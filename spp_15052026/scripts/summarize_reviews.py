"""
summarize_reviews.py
--------------------
Called by review_analysis.py to summarize reviews flagged with needs_summary=True.

Receives the full text-only reviews DataFrame (all reviews, not just long ones).
Only reviews where needs_summary=True are passed through the summarizer.
All others are written to output unchanged.

The summarizer backend is BART (philschmid/bart-large-cnn-samsum) by default.
To swap backends, replace the _summarize_with_bart() call in _run_backend()
with any alternative (e.g. _summarize_with_api()).

Output parquet columns added:
    - summary         : summarized text for long reviews; original trans_review otherwise
    - was_summarized  : True if the summary model was applied, False otherwise

Usage (called from review_analysis.py):
    from summarize_reviews import summarize_reviews
    summarize_reviews(reviews_flagged, output_dir="./data")

Can also be run standalone:
    python summarize_reviews.py \
        --input  data/reviews_flagged.parquet \
        --output data/reviews_summarized.parquet
"""

import argparse
import gc
import os
import sys

import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_MODEL   = "philschmid/bart-large-cnn-samsum"
GPU_BATCH_SIZE  = 32
MAX_INPUT_CHARS = 3000      # reviews beyond this are chunked before BART sees them
                             # BART hard limit is ~1024 tokens ≈ 750 words ≈ 4000 chars
                             # Using 3000 to stay safely under and preserve tail content

# ---------------------------------------------------------------------------
# Backend: BART
# ---------------------------------------------------------------------------

def _load_bart(model_name: str = DEFAULT_MODEL):
    """
    Load the BART model and tokenizer directly.
    Avoids relying on the 'summarization' pipeline task which was removed
    in newer versions of transformers.
    Returns (model, tokenizer, device).
    """
    try:
        import torch
        from transformers import BartForConditionalGeneration, BartTokenizer
    except ImportError:
        raise ImportError("transformers and torch are required: pip install transformers torch")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype  = torch.float16 if device.type == "cuda" else torch.float32

    print(f"  Loading model : {model_name}")
    print(f"  Device        : {'GPU' if device.type == 'cuda' else 'CPU'}")

    tokenizer = BartTokenizer.from_pretrained(model_name)
    model     = BartForConditionalGeneration.from_pretrained(model_name, torch_dtype=dtype)
    model     = model.to(device)
    model.eval()

    return model, tokenizer, device


def _bart_summarize_text(text: str, model, tokenizer, device) -> str:
    """
    Summarize a single review text using BART.

    Handles reviews longer than MAX_INPUT_CHARS by splitting into chunks,
    summarizing each chunk independently, then summarizing the combined
    chunk summaries. This avoids silent truncation of long reviews.
    """
    import torch

    text = " ".join(text.split())   # normalise whitespace

    chunks = [text] if len(text) <= MAX_INPUT_CHARS else _split_into_chunks(text, MAX_INPUT_CHARS)

    def _generate(chunk):
        words   = len(chunk.split())
        max_len = min(80, max(30, words // 2))
        min_len = max(15, max_len // 2)
        try:
            inputs = tokenizer(
                chunk,
                return_tensors="pt",
                max_length=1024,
                truncation=True,
            ).to(device)
            with torch.no_grad():
                ids = model.generate(
                    **inputs,
                    max_length=max_len,
                    min_length=min_len,
                    do_sample=False,
                    num_beams=4,
                )
            return tokenizer.decode(ids[0], skip_special_tokens=True).strip()
        except Exception as e:
            print(f"  [WARN] BART chunk error: {e} — keeping original chunk")
            return chunk

    chunk_summaries = [_generate(c) for c in chunks]

    if len(chunk_summaries) == 1:
        return chunk_summaries[0]

    # Merge chunk summaries into a single coherent summary
    combined = " ".join(chunk_summaries)
    words    = len(combined.split())
    max_len  = min(120, max(40, words // 2))
    min_len  = max(20, max_len // 2)

    try:
        inputs = tokenizer(
            combined,
            return_tensors="pt",
            max_length=1024,
            truncation=True,
        ).to(device)
        with torch.no_grad():
            ids = model.generate(
                **inputs,
                max_length=max_len,
                min_length=min_len,
                do_sample=False,
                num_beams=4,
            )
        return tokenizer.decode(ids[0], skip_special_tokens=True).strip()
    except Exception as e:
        print(f"  [WARN] BART merge summary error: {e} — concatenating chunk summaries")
        return combined


def _split_into_chunks(text: str, max_chars: int) -> list:
    """
    Split text into chunks of at most max_chars, preferring sentence boundaries.
    """
    import re
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks, current = [], ""

    for sentence in sentences:
        if len(current) + len(sentence) + 1 <= max_chars:
            current = (current + " " + sentence).strip()
        else:
            if current:
                chunks.append(current)
            # If a single sentence exceeds max_chars, hard-split it
            if len(sentence) > max_chars:
                for i in range(0, len(sentence), max_chars):
                    chunks.append(sentence[i:i + max_chars])
                current = ""
            else:
                current = sentence

    if current:
        chunks.append(current)

    return chunks if chunks else [text]


def _fill_missing_trans_reviews(reviews: pd.DataFrame) -> pd.DataFrame:
    """
    Fill empty trans_review values from org_review when the translation field
    is missing but the original review text is present.
    """
    if "trans_review" not in reviews.columns:
        raise ValueError("Text column 'trans_review' not found in reviews DataFrame.")
    if "org_review" not in reviews.columns:
        return reviews

    reviews = reviews.copy()
    mask = reviews["trans_review"].isna() | reviews["trans_review"].astype(str).str.strip().eq("")
    if mask.any():
        reviews.loc[mask, "trans_review"] = reviews.loc[mask, "org_review"]
        print(f"  Filled trans_review from org_review for {int(mask.sum()):,} rows")
    return reviews


def _summarize_with_bart(texts: list, model_name: str = DEFAULT_MODEL) -> list:
    """
    Summarize a list of texts using BART.
    Returns a list of summary strings in the same order.
    """
    import torch

    model, tokenizer, device = _load_bart(model_name)
    summaries = []

    for i, text in enumerate(texts):
        if i % 100 == 0:
            print(f"    Summarizing review {i+1}/{len(texts)} ...")
        summaries.append(_bart_summarize_text(text, model, tokenizer, device))

        # Periodically free GPU memory
        if i % 500 == 0 and device.type == "cuda":
            torch.cuda.empty_cache()
            gc.collect()

    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return summaries


# ---------------------------------------------------------------------------
# Swap point: alternative backends
# ---------------------------------------------------------------------------

def _summarize_with_api(texts: list, **kwargs) -> list:
    """
    Placeholder for an API-based summarizer (e.g. Claude via Anthropic API).
    Replace the body of this function to switch backends.

    Should return a list of summary strings in the same order as `texts`.
    """
    raise NotImplementedError(
        "API backend not yet implemented. "
        "Add your summarization logic here and call _run_backend() with backend='api'."
    )


def _run_backend(texts: list, backend: str = "bart", **kwargs) -> list:
    """
    Route to the selected summarization backend.
    Add new backends here as elif branches.
    """
    if backend == "bart":
        return _summarize_with_bart(texts, **kwargs)
    elif backend == "api":
        return _summarize_with_api(texts, **kwargs)
    else:
        raise ValueError(f"Unknown backend: '{backend}'. Choose 'bart' or 'api'.")


# ---------------------------------------------------------------------------
# Main entry point (called by review_analysis.py)
# ---------------------------------------------------------------------------

def _load_existing_output(output_path: str, id_col: str) -> set:
    """
    Load already-summarized review IDs from the output parquet if it exists.
    Returns a set of IDs, or an empty set if the file does not exist.
    """
    if not os.path.exists(output_path):
        return set()

    try:
        existing = pd.read_parquet(output_path, columns=[id_col])
        ids = set(existing[id_col].dropna().tolist())
        print(f"  Checkpoint found   : {output_path}")
        print(f"  Already summarized : {len(ids):,} reviews (will be skipped)")
        return ids
    except Exception as e:
        print(f"  [WARN] Could not read existing output ({e}) -- starting fresh.")
        return set()


def summarize_reviews(
    reviews: pd.DataFrame,
    output_dir: str = "./data",
    backend: str = "bart",
    model_name: str = DEFAULT_MODEL,
    text_col: str = "trans_review",
    flag_col: str = "needs_summary",
    id_col: str = "review_id",
    output_filename: str = "reviews_summarized.parquet",
) -> pd.DataFrame:
    """
    Summarize reviews where needs_summary=True, skipping any review_id
    that already appears in the output parquet from a previous run.

    Parameters
    ----------
    reviews         : Full flagged reviews DataFrame from review_analysis.py
    output_dir      : Directory to save the output parquet
    backend         : 'bart' (default) or 'api'
    model_name      : Model identifier (used by bart backend)
    text_col        : Column containing review text to summarize
    flag_col        : Boolean column marking which reviews need summarization
    id_col          : Column used to identify already-processed reviews
    output_filename : Output parquet filename

    Returns
    -------
    pd.DataFrame with 'summary' and 'was_summarized' columns added.
    """
    if text_col not in reviews.columns:
        raise ValueError(f"Text column '{text_col}' not found in reviews DataFrame.")
    if flag_col not in reviews.columns:
        raise ValueError(f"Flag column '{flag_col}' not found in reviews DataFrame.")
    if id_col not in reviews.columns:
        raise ValueError(f"ID column '{id_col}' not found in reviews DataFrame.")

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)

    # Load already-processed IDs from the existing output parquet
    already_done = _load_existing_output(output_path, id_col)

    reviews = reviews.copy()
    reviews = _fill_missing_trans_reviews(reviews)

    # Split into: already done / needs summary / pass-through (short)
    if already_done:
        remaining = reviews[~reviews[id_col].isin(already_done)].copy()
    else:
        remaining = reviews

    to_summarize = remaining[remaining[flag_col] == True].copy()
    pass_through = remaining[remaining[flag_col] == False].copy()

    n_total   = len(reviews)
    n_skipped = n_total - len(remaining)
    n_long    = len(to_summarize)
    n_short   = len(pass_through)

    print("=" * 55)
    print("SUMMARIZATION")
    print("=" * 55)
    print(f"  Total reviews       : {n_total:,}")
    print(f"  Skipped (done)      : {n_skipped:,}")
    print(f"  To summarize (>100) : {n_long:,}")
    print(f"  Pass through (<=100): {n_short:,}")
    print(f"  Backend             : {backend}")
    if backend == "bart":
        print(f"  Model               : {model_name}")
    print()

    # Summarize new long reviews
    if n_long > 0:
        texts     = to_summarize[text_col].fillna("").tolist()
        summaries = _run_backend(texts, backend=backend, model_name=model_name)
        to_summarize["summary"]        = summaries
        to_summarize["was_summarized"] = True
    else:
        print("  No new reviews to summarize.")
        to_summarize["summary"]        = pd.Series(dtype="str")
        to_summarize["was_summarized"] = pd.Series(dtype="bool")

    # Pass-through reviews keep their original text as the summary
    pass_through["summary"]        = pass_through[text_col]
    pass_through["was_summarized"] = False

    # Combine new results with existing output
    new_results = pd.concat([to_summarize, pass_through], ignore_index=True)

    if already_done and os.path.exists(output_path):
        existing_df = pd.read_parquet(output_path)
        result      = pd.concat([existing_df, new_results], ignore_index=True)
    else:
        result = new_results

    result.to_parquet(output_path, index=False)

    print(f"  Saved -> {output_path}")
    print(f"  Total rows in output : {len(result):,}")
    print(f"  New rows added       : {len(new_results):,}")
    print(f"  Columns: {list(result.columns)}")

    return result

# Standalone entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Summarize flagged reviews in a parquet file.")
    parser.add_argument("--input",    "-i", required=True,  help="Path to reviews_flagged.parquet")
    parser.add_argument("--output",   "-o", required=True,  help="Path for output parquet")
    parser.add_argument("--backend",  "-b", default="bart", choices=["bart", "api"],
                        help="Summarization backend (default: bart)")
    parser.add_argument("--model",    "-m", default=DEFAULT_MODEL,
                        help=f"Model name for bart backend (default: {DEFAULT_MODEL})")
    parser.add_argument("--text_col", "-c", default="trans_review",
                        help="Review text column (default: trans_review)")
    parser.add_argument("--flag_col", "-f", default="needs_summary",
                        help="Flag column (default: needs_summary)")
    parser.add_argument("--id_col",   "-id", default="review_id",
                        help="Review ID column used to skip already-processed reviews (default: review_id)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input file not found: {args.input}")
        sys.exit(1)

    print(f"Loading : {args.input}")
    reviews = pd.read_parquet(args.input)
    print(f"  Rows  : {len(reviews):,}\n")

    output_dir      = os.path.dirname(os.path.abspath(args.output))
    output_filename = os.path.basename(args.output)

    summarize_reviews(
        reviews,
        output_dir=output_dir,
        backend=args.backend,
        model_name=args.model,
        text_col=args.text_col,
        flag_col=args.flag_col,
        id_col=args.id_col,
        output_filename=output_filename,
    )


if __name__ == "__main__":
    main()