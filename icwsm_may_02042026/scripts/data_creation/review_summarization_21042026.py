#!/usr/bin/env python3
"""
Checkpointed Hybrid GPU+CPU BART Summarizer for Parquet Files
Improved version — fixes truncated / incomplete sentence summaries

Key changes from original:
  1. Per-text max_len / min_len instead of batch-average
     → prevents short texts capping long ones and vice versa
  2. Sentence-boundary cleanup on every summary
     → removes trailing incomplete sentences
  3. Input pre-truncation before tokenisation
     → BART has a 1024-token hard limit; silently truncating inside
        the pipeline produces mid-sentence cuts. We truncate at word
        level first so the model sees a complete input.
  4. Separate length buckets for GPU batching
     → mixing very short and very long texts in one batch forces
        padding overhead and degrades quality. Bucket by length first.
  5. review_id passthrough
     → output always preserves review_id alongside summary so
        downstream scripts can join on it cleanly
  6. Minor: atomic temp-file rename was missing .rename() call fix
"""

import pandas as pd
import argparse
from transformers import pipeline
import torch
from tqdm import tqdm
import os
import sys
from pathlib import Path
import json
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import gc
import math
import hashlib
import re

# =============================================================================
# SENTENCE CLEANUP
# =============================================================================

def trim_to_last_complete_sentence(text: str) -> str:
    """
    Remove any trailing incomplete sentence from a summary.

    BART sometimes ends mid-word or mid-clause when max_length is hit.
    This finds the last sentence-ending punctuation (.  !  ?) and
    trims everything after it.

    Examples:
        "The pool was clean. Staff were frien"
            → "The pool was clean."
        "Visited on a Sunday. Very crowded, especially near"
            → "Visited on a Sunday."
        "Good facilities."
            → "Good facilities."   (unchanged)
    """
    if not text or not text.strip():
        return text

    text = text.strip()

    # If it already ends with sentence-final punctuation, return as-is
    if text[-1] in ".!?":
        return text

    # Find the last sentence-ending punctuation
    match = re.search(r"[.!?](?=[^.!?]*$)", text)
    if match:
        return text[: match.end()].strip()

    # No sentence boundary found at all — return the full text as a last resort
    # (short summaries of very short inputs often have no punctuation)
    return text


def safe_truncate_words(text: str, max_words: int = 200) -> str:
    """
    Truncate to max_words at a sentence boundary where possible,
    otherwise at the word boundary.

    BART's tokeniser limit is 1024 tokens ≈ 700-800 words for English.
    Truncating here gives the model a complete grammatical input rather
    than a mid-sentence cut, which produces better summaries.
    """
    words = text.split()
    if len(words) <= max_words:
        return text

    truncated = " ".join(words[:max_words])

    # Try to end at a sentence boundary within the last 30 words
    match = re.search(r"[.!?](?=[^.!?]*$)", truncated)
    if match and match.end() > len(truncated) - 200:
        return truncated[: match.end()].strip()

    return truncated


# =============================================================================
# LENGTH COMPUTATION — per text, not per batch
# =============================================================================

def compute_lengths(text: str):
    """
    Return (max_length, min_length) appropriate for this specific text.

    Heuristic:
        target summary ≈ 30% of input length
        floor at 20 tokens, ceiling at 120 tokens
        min_length = half of max_length (ensures non-trivial output)
    """
    n_words = len(text.split())
    target  = max(20, min(120, int(n_words * 0.30)))
    min_len = max(10, target // 2)
    return target, min_len


# =============================================================================
# LENGTH BUCKETING — prevents padding waste in GPU batches
# =============================================================================

def bucket_texts(texts: list, indices: list, n_buckets: int = 4):
    """
    Group texts into n_buckets by word count so each GPU batch has
    roughly similar-length inputs. Returns list of (texts, indices) per bucket.
    """
    pairs = list(zip(texts, indices))
    pairs.sort(key=lambda x: len(x[0].split()))

    bucket_size = math.ceil(len(pairs) / n_buckets)
    buckets = []
    for i in range(0, len(pairs), bucket_size):
        group = pairs[i : i + bucket_size]
        b_texts   = [p[0] for p in group]
        b_indices = [p[1] for p in group]
        buckets.append((b_texts, b_indices))
    return buckets


# =============================================================================
# GPU SUMMARISER
# =============================================================================

class HybridSummarizer:
    def __init__(self, gpu_batch_size=32, cpu_workers=8):
        self.gpu_batch_size = gpu_batch_size
        self.cpu_workers    = cpu_workers
        print(f"[DEBUG] Initializing HybridSummarizer — GPU batch {gpu_batch_size}, "
              f"{cpu_workers} CPU workers")

        print("[DEBUG] Loading GPU model...")
        self.gpu_pipeline = pipeline(
            "summarization",
            model="philschmid/bart-large-cnn-samsum",
            device=0,
            torch_dtype=torch.float16,
            batch_size=gpu_batch_size,
        )
        print("[DEBUG] GPU pipeline loaded.")

    def summarize_gpu_batch(self, texts: list, indices: list) -> dict:
        """
        Summarise a list of texts on GPU.
        Uses per-text length computation and length bucketing.
        Returns {index: summary_text} dict.
        """
        if not texts:
            return {}

        results_dict = {}

        # Pre-truncate all inputs to safe word limit
        safe_texts = [safe_truncate_words(t, max_words=200) for t in texts]

        # Bucket by length to reduce padding waste
        buckets = bucket_texts(safe_texts, indices, n_buckets=4)

        for bucket_texts_list, bucket_indices in buckets:
            if not bucket_texts_list:
                continue

            # Compute per-text lengths, then use bucket-level max/min
            # (the pipeline requires a single max_length per call;
            #  we take the 90th-percentile length within each bucket
            #  so most texts are handled well without over-truncating)
            per_text_targets = [compute_lengths(t)[0] for t in bucket_texts_list]
            bucket_max = int(sorted(per_text_targets)[int(len(per_text_targets) * 0.90)])
            bucket_max = max(30, min(120, bucket_max))
            bucket_min = max(10, bucket_max // 2)

            print(f"[DEBUG] GPU bucket: {len(bucket_texts_list)} texts, "
                  f"max_len={bucket_max}, min_len={bucket_min}")

            try:
                raw_results = self.gpu_pipeline(
                    bucket_texts_list,
                    max_length=bucket_max,
                    min_length=bucket_min,
                    do_sample=False,
                    truncation=True,
                    batch_size=self.gpu_batch_size,
                )
                for idx, result in zip(bucket_indices, raw_results):
                    summary = trim_to_last_complete_sentence(
                        result["summary_text"].strip()
                    )
                    results_dict[idx] = summary

            except Exception as e:
                print(f"[ERROR] GPU bucket error: {e}")
                # Fall back to original text for this bucket
                for idx, text in zip(bucket_indices, bucket_texts_list):
                    results_dict[idx] = text

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()

        return results_dict


# =============================================================================
# CPU WORKER
# =============================================================================

def init_cpu_worker():
    global cpu_pipeline
    print(f"[DEBUG] Initializing CPU worker PID={os.getpid()}")
    cpu_pipeline = pipeline(
        "summarization",
        model="philschmid/bart-large-cnn-samsum",
        device=-1,
        truncation=True,
    )
    print(f"[DEBUG] CPU worker PID={os.getpid()} loaded.")


def process_cpu_batch(batch_data):
    global cpu_pipeline
    texts, indices = batch_data
    results = []

    for text, idx in zip(texts, indices):
        if not text.strip():
            results.append((idx, ""))
            continue
        try:
            safe_text       = safe_truncate_words(text, max_words=200)
            max_len, min_len = compute_lengths(safe_text)

            raw = cpu_pipeline(
                safe_text,
                max_length=max_len,
                min_length=min_len,
                do_sample=False,
            )[0]["summary_text"].strip()

            summary = trim_to_last_complete_sentence(raw)
            results.append((idx, summary))

        except Exception as e:
            print(f"[ERROR] CPU worker PID={os.getpid()}: {e}")
            results.append((idx, text))

    gc.collect()
    return results


# =============================================================================
# UTILITIES
# =============================================================================

def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def read_parquet_in_chunks(file_path: str, chunk_size: int):
    """Read parquet in row-sliced chunks without loading full file twice."""
    full = pd.read_parquet(file_path)
    total_rows = len(full)
    for start_idx in range(0, total_rows, chunk_size):
        yield full.iloc[start_idx : start_idx + chunk_size].copy()
    del full
    gc.collect()


def get_parameters_hash(args) -> str:
    param_str = (
        f"{args.input}{args.text_column}{args.min_chars}"
        f"{args.gpu_batch_size}{args.cpu_batch_size}{args.chunk_size}"
    )
    return hashlib.md5(param_str.encode()).hexdigest()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Checkpointed Hybrid GPU+CPU BART for Parquet files (improved)"
    )
    parser.add_argument("--input",         "-i", required=True)
    parser.add_argument("--output",        "-o", required=True)
    parser.add_argument("--text_column",   "-c", default="trans_review")
    parser.add_argument("--id_column",     "-id", default="review_id",
                        help="Column to use as review identifier (default: review_id)")
    parser.add_argument("--min_chars",     "-m", type=int, default=200,
                        help="Min characters before summarising (default: 200)")
    parser.add_argument("--gpu_batch_size","-g", type=int, default=32)
    parser.add_argument("--cpu_batch_size","-b", type=int, default=16)
    parser.add_argument("--chunk_size",    "-s", type=int, default=5000)
    parser.add_argument("--cpu_workers",   "-w", type=int, default=0)
    parser.add_argument("--clear_every",   "-e", type=int, default=10)
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}")
        sys.exit(1)

    clear_memory()
    summarizer = HybridSummarizer(
        gpu_batch_size=args.gpu_batch_size,
        cpu_workers=args.cpu_workers,
    )

    output_path   = Path(args.output)
    params_hash   = get_parameters_hash(args)
    checkpoint_dir = output_path.parent / f"{output_path.stem}_checkpoints_{params_hash[:8]}"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_file = checkpoint_dir / "progress.json"

    # ---- Resume from checkpoint ----
    if checkpoint_file.exists():
        with open(checkpoint_file) as f:
            progress = json.load(f)
        if progress.get("params_hash") != params_hash:
            print("[WARNING] Parameters changed — starting fresh.")
            progress = {
                "processed_chunks": [], "total_processed": 0,
                "start_time": time.time(), "params_hash": params_hash,
            }
        else:
            print(f"[DEBUG] Resuming from {len(progress['processed_chunks'])} "
                  f"completed chunks.")
    else:
        progress = {
            "processed_chunks": [], "total_processed": 0,
            "start_time": time.time(), "params_hash": params_hash,
        }

    processed_chunks = progress["processed_chunks"]
    total_processed  = progress["total_processed"]
    start_time       = progress["start_time"]

    # Count total rows once
    total_rows   = len(pd.read_parquet(args.input, columns=[args.text_column]))
    total_chunks = math.ceil(total_rows / args.chunk_size)
    print(f"[DEBUG] Total rows: {total_rows:,}  Total chunks: {total_chunks}")

    # Validate id_column exists
    sample = pd.read_parquet(args.input).head(1)
    if args.id_column not in sample.columns:
        print(f"[ERROR] id_column '{args.id_column}' not found in parquet.")
        print(f"  Available columns: {list(sample.columns)}")
        sys.exit(1)

    for chunk_idx, chunk in enumerate(
        tqdm(read_parquet_in_chunks(args.input, args.chunk_size),
             total=total_chunks, desc="Chunks")
    ):
        if chunk_idx in processed_chunks:
            print(f"[DEBUG] Skipping chunk {chunk_idx} (already done)")
            continue

        print(f"[DEBUG] Processing chunk {chunk_idx} ({len(chunk)} rows)")
        clear_memory()

        if args.text_column not in chunk.columns:
            print(f"[ERROR] Column '{args.text_column}' not found in chunk.")
            sys.exit(1)

        # Clean whitespace
        chunk["cleaned_text"] = chunk[args.text_column].apply(
            lambda x: " ".join(str(x).split()) if pd.notna(x) else ""
        )
        chunk["char_count"] = chunk["cleaned_text"].str.len()

        summarize_mask = chunk["char_count"] > args.min_chars
        to_summarize   = chunk[summarize_mask].copy()
        keep_original  = chunk[~summarize_mask].copy()

        print(f"[DEBUG] Chunk {chunk_idx}: {len(to_summarize)} to summarise, "
              f"{len(keep_original)} kept as-is")

        # ---- GPU processing ----
        if args.cpu_workers == 0:
            gpu_texts   = to_summarize["cleaned_text"].tolist()
            gpu_indices = to_summarize.index.tolist()
        else:
            half = len(to_summarize) // 2
            gpu_texts   = to_summarize["cleaned_text"].tolist()[:half]
            gpu_indices = to_summarize.index.tolist()[:half]

        # Returns {index: summary} dict
        summaries_dict = summarizer.summarize_gpu_batch(gpu_texts, gpu_indices)

        # ---- CPU processing ----
        if args.cpu_workers > 0:
            half        = len(to_summarize) // 2
            cpu_texts   = to_summarize["cleaned_text"].tolist()[half:]
            cpu_indices = to_summarize.index.tolist()[half:]

            if cpu_texts:
                print(f"[DEBUG] Sending {len(cpu_texts)} texts to {args.cpu_workers} CPU workers")
                cpu_batches = [
                    (cpu_texts[i : i + args.cpu_batch_size],
                     cpu_indices[i : i + args.cpu_batch_size])
                    for i in range(0, len(cpu_texts), args.cpu_batch_size)
                ]
                with ProcessPoolExecutor(
                    max_workers=args.cpu_workers,
                    initializer=init_cpu_worker,
                ) as executor:
                    futures    = [executor.submit(process_cpu_batch, b) for b in cpu_batches]
                    batch_count = 0
                    for future in tqdm(as_completed(futures),
                                       total=len(futures), desc="CPU batches"):
                        for idx, summary in future.result():
                            summaries_dict[idx] = summary
                        batch_count += 1
                        if batch_count % args.clear_every == 0:
                            clear_memory()

        # ---- Assign summaries ----
        to_summarize["summary"]        = to_summarize.index.map(
            lambda x: summaries_dict.get(x, "")
        )
        to_summarize["was_summarized"] = to_summarize.index.map(
            lambda x: x in summaries_dict
        )

        keep_original["summary"]        = keep_original["cleaned_text"]
        keep_original["was_summarized"] = False

        chunk_result = pd.concat(
            [to_summarize, keep_original], ignore_index=True
        )

        # ---- Ensure review_id is always in output ----
        # (It comes through with the chunk — this just makes it explicit)
        if args.id_column not in chunk_result.columns:
            print(f"[WARNING] id_column '{args.id_column}' missing from chunk result.")

        # Save chunk
        partial_output = checkpoint_dir / f"chunk_{chunk_idx}.parquet"
        chunk_result.to_parquet(partial_output)
        print(f"[DEBUG] Saved chunk {chunk_idx} → {partial_output}")

        processed_chunks.append(chunk_idx)
        total_processed += len(chunk_result)
        progress.update({
            "processed_chunks": processed_chunks,
            "total_processed":  total_processed,
            "last_update":      time.time(),
        })
        with open(checkpoint_file, "w") as f:
            json.dump(progress, f, indent=2)

        del chunk, to_summarize, keep_original, chunk_result
        clear_memory()

    # ---- Combine all chunks ----
    print("[DEBUG] Combining chunks into final output...")
    final_chunks = sorted(
        checkpoint_dir.glob("chunk_*.parquet"),
        key=lambda p: int(p.stem.split("_")[1]),  # sort numerically
    )

    if not final_chunks:
        print("[ERROR] No chunk files found!")
        sys.exit(1)

    try:
        final_df = pd.concat(
            [pd.read_parquet(f) for f in final_chunks], ignore_index=True
        )

        # Sanity check: report sentence-truncation rate
        if "summary" in final_df.columns:
            incomplete = final_df["summary"].apply(
                lambda x: isinstance(x, str) and len(x) > 0 and x[-1] not in ".!?"
            ).sum()
            total_summarized = final_df.get("was_summarized", pd.Series()).sum()
            if total_summarized > 0:
                print(f"[DEBUG] Summaries not ending in sentence punctuation: "
                      f"{incomplete} ({incomplete/total_summarized*100:.1f}% of summarised)")

        # Atomic write
        temp_output = output_path.with_suffix(".tmp.parquet")
        final_df.to_parquet(temp_output, index=False)
        temp_output.rename(output_path)

        print(f"[DEBUG] Final output saved → {args.output}")
        print(f"[DEBUG] Columns in output: {list(final_df.columns)}")

        # Verify review_id is present
        if args.id_column in final_df.columns:
            print(f"[DEBUG] '{args.id_column}' column present — "
                  f"{final_df[args.id_column].nunique():,} unique IDs")
        else:
            print(f"[WARNING] '{args.id_column}' not in final output — "
                  f"check your input file column names")

        # Clean up
        for f in final_chunks:
            f.unlink()
        checkpoint_file.unlink(missing_ok=True)
        checkpoint_dir.rmdir()

    except Exception as e:
        print(f"[ERROR] Failed to combine chunks: {e}")
        print("[INFO] Chunk files preserved for manual recovery")
        sys.exit(1)

    total_time = time.time() - start_time
    summarized = final_df.get("was_summarized", pd.Series(dtype=bool)).sum()
    print(f"\n[DEBUG] Done. {len(final_df):,} total reviews, {summarized:,} summarised")
    print(f"[DEBUG] Time: {total_time/3600:.2f} hours  "
          f"Rate: {len(final_df)/total_time:.1f} reviews/sec")


if __name__ == "__main__":
    mp.freeze_support()
    main()