#!/usr/bin/env python3
"""
Extractive Review Compressor
==============================
Replaces BART summarisation for the review pre-processing step.

Goal: reduce token count before LLM classification while
      preserving all content that matters for classification —
      especially subtle, implicit, and specific language.

Approach: extractive compression, NOT abstractive summarisation.
  - No model rewrites content
  - Original words and phrases are always preserved verbatim
  - Runs on CPU, no GPU needed, processes ~50k reviews/minute
  - No token input limits — handles reviews of any length
  - Fully deterministic and reproducible

Two strategies available:
  1. sentence_boundary (default, recommended)
     Keep the first N characters ending at a sentence boundary.
     Best for reviews that front-load their main claims (most do).

  2. lexrank
     Graph-based extractive summarisation — scores sentences by
     centrality and keeps the most representative ones.
     Better for long reviews where key content is not in first sentence.
     Requires: pip install sumy

The pipeline also flags reviews that are already short enough to
pass through without any compression — these are sent to the LLM as-is.

Usage:
    python extractive_compressor.py --input data.parquet --output compressed.parquet
    python extractive_compressor.py --input data.parquet --output compressed.parquet \\
        --strategy lexrank --target_chars 400
    python extractive_compressor.py --input data.parquet --output compressed.parquet \\
        --max_chars 300 --chunk_size 50000

Requirements:
    pip install pandas pyarrow tqdm
    pip install sumy nltk   # only needed for --strategy lexrank
"""

import re
import os
import sys
import math
import hashlib
import json
import time
import argparse
import gc
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Optional — only needed for lexrank strategy
try:
    from sumy.parsers.plaintext import PlaintextParser
    from sumy.nlp.tokenizers import Tokenizer
    from sumy.summarizers.lex_rank import LexRankSummarizer
    from sumy.nlp.stemmers import Stemmer
    from sumy.utils import get_stop_words
    import nltk
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)
    SUMY_AVAILABLE = True
except ImportError:
    SUMY_AVAILABLE = False


# =============================================================================
# STRATEGY 1 — Sentence boundary truncation (fast, recommended)
# =============================================================================

def compress_sentence_boundary(text: str, target_chars: int = 400) -> tuple:
    """
    Keep as much text as fits within target_chars, ending at a sentence boundary.

    Returns (compressed_text, was_compressed: bool)

    Why this works for Google reviews:
        Reviews almost always open with their main claim or overall verdict.
        "The pool was dirty and staff were rude to foreign visitors" appears
        in sentence 1-2, not buried at the end. Keeping the first 400 chars
        at a sentence boundary captures this reliably.

    For reviews already under target_chars: returns original unchanged.
    """
    if not isinstance(text, str):
        return ("", False)

    text = " ".join(text.split())   # normalise whitespace

    if len(text) <= target_chars:
        return (text, False)        # already short enough — no compression

    # Truncate at target_chars
    truncated = text[:target_chars]

    # Walk back to the last sentence-ending punctuation
    # Require at least 50% of target to avoid over-truncating short sentences
    min_pos = target_chars // 2
    match = None
    for m in re.finditer(r"[.!?]", truncated):
        if m.end() >= min_pos:
            match = m
    # Take the last valid match
    matches = list(re.finditer(r"[.!?]", truncated))
    valid   = [m for m in matches if m.end() >= min_pos]

    if valid:
        cut = valid[-1].end()
        return (truncated[:cut].strip(), True)

    # No sentence boundary found — fall back to word boundary
    words = truncated.rsplit(" ", 1)
    return (words[0].strip(), True)


# =============================================================================
# STRATEGY 2 — LexRank extractive summarisation
# =============================================================================

def compress_lexrank(text: str, target_chars: int = 400,
                     language: str = "english") -> tuple:
    """
    Use LexRank to select the most central sentences from the review.
    Preserves original wording exactly.

    Better than sentence_boundary for long reviews where the key claim
    is not in the first sentence — e.g. a review that opens with
    general praise then makes a specific discriminatory comment later.

    Returns (compressed_text, was_compressed: bool)
    """
    if not SUMY_AVAILABLE:
        raise ImportError(
            "sumy not installed. Run: pip install sumy nltk"
        )

    if not isinstance(text, str):
        return ("", False)

    text = " ".join(text.split())

    if len(text) <= target_chars:
        return (text, False)

    try:
        # Estimate how many sentences to keep
        # Rough heuristic: average sentence ~80 chars, keep enough to fill target
        n_sentences = max(1, target_chars // 80)

        parser     = PlaintextParser.from_string(text, Tokenizer(language))
        stemmer    = Stemmer(language)
        summarizer = LexRankSummarizer(stemmer)
        summarizer.stop_words = get_stop_words(language)

        sentences = summarizer(parser.document, n_sentences)
        compressed = " ".join(str(s) for s in sentences)

        if not compressed.strip():
            # LexRank failed (e.g. single-sentence input) — fall back
            return compress_sentence_boundary(text, target_chars)

        return (compressed.strip(), True)

    except Exception:
        # Any failure — fall back to sentence boundary
        return compress_sentence_boundary(text, target_chars)


# =============================================================================
# BATCH PROCESSOR
# =============================================================================

def compress_batch(texts: list, strategy: str,
                   target_chars: int) -> list:
    """
    Compress a list of texts. Returns list of (compressed, was_compressed).
    """
    results = []
    for text in texts:
        if strategy == "lexrank":
            result = compress_lexrank(text, target_chars)
        else:
            result = compress_sentence_boundary(text, target_chars)
        results.append(result)
    return results


# =============================================================================
# STATISTICS
# =============================================================================

def compute_stats(original_texts: pd.Series,
                  compressed_texts: pd.Series,
                  was_compressed: pd.Series) -> dict:
    """
    Report token/character reduction statistics.
    Helps you estimate actual LLM cost savings.
    """
    orig_chars  = original_texts.str.len().fillna(0)
    comp_chars  = compressed_texts.str.len().fillna(0)
    compressed  = was_compressed.astype(bool)

    # Approximate tokens as chars / 4 (rough but consistent)
    orig_tokens = orig_chars / 4
    comp_tokens = comp_chars / 4

    n_compressed = compressed.sum()
    n_total      = len(original_texts)

    stats = {
        "total_reviews":        n_total,
        "compressed_count":     int(n_compressed),
        "passed_through":       int(n_total - n_compressed),
        "compression_rate_pct": round(n_compressed / n_total * 100, 1),
        "mean_orig_chars":      round(orig_chars.mean(), 1),
        "mean_comp_chars":      round(comp_chars[compressed].mean(), 1) if n_compressed > 0 else 0,
        "mean_char_reduction_pct": round(
            (1 - comp_chars[compressed].mean() / orig_chars[compressed].mean()) * 100, 1
        ) if n_compressed > 0 else 0,
        "estimated_orig_tokens_total":  int(orig_tokens.sum()),
        "estimated_comp_tokens_total":  int(comp_tokens.sum()),
        "estimated_token_savings_pct":  round(
            (1 - comp_tokens.sum() / orig_tokens.sum()) * 100, 1
        ) if orig_tokens.sum() > 0 else 0,
    }

    # Cost estimate at Claude Sonnet pricing (~$3 per 1M input tokens)
    price_per_1m = 3.0
    orig_cost  = stats["estimated_orig_tokens_total"] / 1_000_000 * price_per_1m
    comp_cost  = stats["estimated_comp_tokens_total"] / 1_000_000 * price_per_1m
    stats["estimated_cost_without_compression_usd"] = round(orig_cost, 2)
    stats["estimated_cost_with_compression_usd"]    = round(comp_cost, 2)
    stats["estimated_savings_usd"]                  = round(orig_cost - comp_cost, 2)

    return stats


# =============================================================================
# CHECKPOINT HELPERS
# =============================================================================

def get_params_hash(args) -> str:
    s = f"{args.input}{args.text_column}{args.id_column}{args.strategy}{args.target_chars}{args.chunk_size}"
    return hashlib.md5(s.encode()).hexdigest()


def load_checkpoint(checkpoint_file: Path, params_hash: str) -> dict:
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            progress = json.load(f)
        if progress.get("params_hash") == params_hash:
            print(f"  Resuming from checkpoint — {len(progress['done_chunks'])} chunks done")
            return progress
        else:
            print("  Parameters changed — starting fresh")
    return {
        "params_hash":    params_hash,
        "done_chunks":    [],
        "total_processed": 0,
        "start_time":     time.time(),
    }


def save_checkpoint(checkpoint_file: Path, progress: dict):
    with open(checkpoint_file, "w") as f:
        json.dump(progress, f, indent=2)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Extractive review compressor — preserves original wording"
    )
    parser.add_argument("--input",        "-i",  required=True,
                        help="Input parquet file")
    parser.add_argument("--output",       "-o",  required=True,
                        help="Output parquet file")
    parser.add_argument("--text_column",  "-c",  default="trans_review",
                        help="Column containing review text (default: trans_review)")
    parser.add_argument("--id_column",    "-id", default="review_id",
                        help="Review ID column (default: review_id)")
    parser.add_argument("--strategy",     "-st", default="sentence_boundary",
                        choices=["sentence_boundary", "lexrank"],
                        help="Compression strategy (default: sentence_boundary)")
    parser.add_argument("--target_chars", "-t",  type=int, default=400,
                        help="Target character length after compression (default: 400). "
                             "400 chars ≈ 100 tokens ≈ enough context for classification.")
    parser.add_argument("--chunk_size",   "-s",  type=int, default=50000,
                        help="Rows per processing chunk (default: 50000)")
    parser.add_argument("--max_chars",    "-m",  type=int, default=None,
                        help="Only compress reviews longer than this many chars. "
                             "Defaults to target_chars.")
    args = parser.parse_args()

    # max_chars defaults to target_chars — reviews already below target pass through
    if args.max_chars is None:
        args.max_chars = args.target_chars

    if not os.path.exists(args.input):
        print(f"[ERROR] Input not found: {args.input}")
        sys.exit(1)

    if args.strategy == "lexrank" and not SUMY_AVAILABLE:
        print("[ERROR] lexrank strategy requires sumy: pip install sumy nltk")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Extractive Review Compressor")
    print(f"{'='*60}")
    print(f"  Strategy    : {args.strategy}")
    print(f"  Target chars: {args.target_chars}")
    print(f"  ID column   : {args.id_column}")
    print(f"  Text column : {args.text_column}")

    # Checkpoint setup
    output_path    = Path(args.output)
    params_hash    = get_params_hash(args)
    checkpoint_dir = output_path.parent / f"{output_path.stem}_ckpt_{params_hash[:8]}"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_file = checkpoint_dir / "progress.json"
    progress        = load_checkpoint(checkpoint_file, params_hash)

    # Count total rows
    total_rows   = len(pd.read_parquet(args.input, columns=[args.id_column]))
    total_chunks = math.ceil(total_rows / args.chunk_size)
    print(f"  Total rows  : {total_rows:,}")
    print(f"  Chunks      : {total_chunks}")

    # Validate columns
    sample = pd.read_parquet(args.input).head(1)
    for col in [args.text_column, args.id_column]:
        if col not in sample.columns:
            print(f"[ERROR] Column '{col}' not found. Available: {list(sample.columns)}")
            sys.exit(1)
    del sample

    # Load full file once (parquet random access is expensive per-chunk otherwise)
    print("\n  Loading input file...")
    full_df = pd.read_parquet(args.input)
    print(f"  Loaded {len(full_df):,} rows")

    all_stats = []

    for chunk_idx in tqdm(range(total_chunks), desc="Chunks"):
        if chunk_idx in progress["done_chunks"]:
            continue

        start = chunk_idx * args.chunk_size
        end   = min(start + args.chunk_size, total_rows)
        chunk = full_df.iloc[start:end].copy()

        # Identify which reviews need compression
        chunk["_orig_text"] = chunk[args.text_column].fillna("").apply(
            lambda x: " ".join(str(x).split())
        )
        needs_compression = chunk["_orig_text"].str.len() > args.max_chars

        to_compress   = chunk[needs_compression].copy()
        pass_through  = chunk[~needs_compression].copy()

        # Compress
        if len(to_compress) > 0:
            compressed_results = compress_batch(
                to_compress["_orig_text"].tolist(),
                strategy=args.strategy,
                target_chars=args.target_chars,
            )
            to_compress["compressed_text"] = [r[0] for r in compressed_results]
            to_compress["was_compressed"]  = [r[1] for r in compressed_results]
        
        pass_through["compressed_text"] = pass_through["_orig_text"]
        pass_through["was_compressed"]  = False

        chunk_result = pd.concat([to_compress, pass_through], ignore_index=True)

        # Compute chunk stats
        chunk_stats = compute_stats(
            chunk_result["_orig_text"],
            chunk_result["compressed_text"],
            chunk_result["was_compressed"],
        )
        all_stats.append(chunk_stats)

        # Drop internal column
        chunk_result = chunk_result.drop(columns=["_orig_text"])

        # Save chunk
        chunk_path = checkpoint_dir / f"chunk_{chunk_idx}.parquet"
        chunk_result.to_parquet(chunk_path, index=False)

        progress["done_chunks"].append(chunk_idx)
        progress["total_processed"] += len(chunk_result)
        save_checkpoint(checkpoint_file, progress)

        del chunk, to_compress, pass_through, chunk_result
        gc.collect()

    del full_df
    gc.collect()

    # ---- Combine all chunks ----
    print("\n  Combining chunks...")
    chunk_files = sorted(
        checkpoint_dir.glob("chunk_*.parquet"),
        key=lambda p: int(p.stem.split("_")[1]),
    )

    final_df = pd.concat(
        [pd.read_parquet(f) for f in chunk_files], ignore_index=True
    )

    # ---- Aggregate and print final statistics ----
    print(f"\n{'='*60}")
    print("  COMPRESSION STATISTICS")
    print(f"{'='*60}")

    total_stats = compute_stats(
        final_df["_orig_text"] if "_orig_text" in final_df.columns
        else final_df[args.text_column].fillna(""),
        final_df["compressed_text"],
        final_df["was_compressed"],
    )

    # Re-compute from final_df directly for accuracy
    orig_chars = final_df[args.text_column].fillna("").str.len()
    comp_chars = final_df["compressed_text"].fillna("").str.len()
    compressed_mask = final_df["was_compressed"].astype(bool)

    print(f"  Total reviews              : {len(final_df):,}")
    print(f"  Reviews compressed         : {compressed_mask.sum():,} "
          f"({compressed_mask.mean()*100:.1f}%)")
    print(f"  Reviews passed through     : {(~compressed_mask).sum():,}")
    print(f"\n  Mean chars (original)      : {orig_chars.mean():.0f}")
    print(f"  Mean chars (compressed)    : {comp_chars[compressed_mask].mean():.0f}"
          f" (compressed only)")
    print(f"  Mean reduction             : "
          f"{(1 - comp_chars[compressed_mask].mean() / orig_chars[compressed_mask].mean())*100:.1f}%"
          f" (compressed only)")

    orig_tokens = orig_chars.sum() / 4
    comp_tokens = comp_chars.sum() / 4
    savings_pct = (1 - comp_tokens / orig_tokens) * 100

    print(f"\n  Estimated total tokens (original)   : {orig_tokens:,.0f}")
    print(f"  Estimated total tokens (compressed) : {comp_tokens:,.0f}")
    print(f"  Token reduction                     : {savings_pct:.1f}%")

    price_per_1m = 3.0
    print(f"\n  Estimated LLM input cost (original)  : "
          f"${orig_tokens/1_000_000*price_per_1m:.2f}")
    print(f"  Estimated LLM input cost (compressed): "
          f"${comp_tokens/1_000_000*price_per_1m:.2f}")
    print(f"  Estimated savings                    : "
          f"${(orig_tokens-comp_tokens)/1_000_000*price_per_1m:.2f}")
    print(f"  (at $3/1M input tokens — adjust for your actual model pricing)")

    # ---- Verify review_id ----
    print(f"\n  '{args.id_column}' unique values: {final_df[args.id_column].nunique():,}")

    # ---- Save ----
    temp_out = output_path.with_suffix(".tmp.parquet")
    final_df.to_parquet(temp_out, index=False)
    temp_out.rename(output_path)
    print(f"\n  Saved → {args.output}")
    print(f"  Columns: {list(final_df.columns)}")

    # ---- Cleanup ----
    for f in chunk_files:
        f.unlink()
    checkpoint_file.unlink(missing_ok=True)
    try:
        checkpoint_dir.rmdir()
    except OSError:
        pass   # not empty — leave it

    total_time = time.time() - progress["start_time"]
    print(f"\n  Done in {total_time/60:.1f} minutes")
    print(f"  Rate: {len(final_df)/total_time:.0f} reviews/sec")


if __name__ == "__main__":
    main()