#!/usr/bin/env python3
"""
Checkpointed Hybrid GPU+CPU BART Summarizer for Parquet Files
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

# ---------------- GPU Summarizer ----------------
class HybridSummarizer:
    def __init__(self, gpu_batch_size=32, cpu_workers=8):
        self.gpu_batch_size = gpu_batch_size
        self.cpu_workers = cpu_workers
        print(f"[DEBUG] Initializing HybridSummarizer with GPU batch {gpu_batch_size} and {cpu_workers} CPU workers")

        print("[DEBUG] Loading GPU model...")
        self.gpu_pipeline = pipeline(
            "summarization",
            model="philschmid/bart-large-cnn-samsum",
            device=0,  # GPU
            torch_dtype=torch.float16,
            batch_size=gpu_batch_size
        )
        print("[DEBUG] GPU pipeline loaded.")

    def summarize_gpu_batch(self, texts):
        """Summarize a list of texts on GPU with batching"""
        if not texts:
            return []

        try:
            avg_words = sum(len(t.split()) for t in texts) // len(texts)
            max_len = min(80, max(30, avg_words // 2))
            min_len = max(15, max_len // 2)

            print(f"[DEBUG] GPU summarizing batch of {len(texts)} texts (max_len={max_len}, min_len={min_len})")
            results = self.gpu_pipeline(
                texts,
                max_length=max_len,
                min_length=min_len,
                do_sample=False,
                truncation=True,
                batch_size=self.gpu_batch_size
            )

            summaries = [r['summary_text'].strip() for r in results]
            print(f"[DEBUG] GPU batch completed, produced {len(summaries)} summaries")
            return summaries

        except Exception as e:
            print(f"[ERROR] GPU batch error: {e}")
            return texts
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            gc.collect()

# ---------------- CPU Worker ----------------
def init_cpu_worker():
    global cpu_pipeline
    print(f"[DEBUG] Initializing CPU worker PID={os.getpid()}")
    cpu_pipeline = pipeline(
        "summarization",
        model="philschmid/bart-large-cnn-samsum",
        device=-1,
        truncation=True
    )
    print(f"[DEBUG] CPU worker PID={os.getpid()} loaded pipeline.")

def process_cpu_batch(batch_data):
    global cpu_pipeline
    texts, indices = batch_data
    results = []

    for text, idx in zip(texts, indices):
        if not text.strip():
            results.append((idx, ""))
            continue
        try:
            input_words = len(text.split())
            max_len = min(60, max(25, input_words // 2))
            summary = cpu_pipeline(
                text,
                max_length=max_len,
                min_length=15,
                do_sample=False
            )[0]['summary_text'].strip()
            results.append((idx, summary))
        except Exception as e:
            print(f"[ERROR] CPU worker PID={os.getpid()} error: {e}")
            results.append((idx, text))
    gc.collect()
    return results

# ---------------- Memory cleanup ----------------
def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# ---------------- Chunk reader ----------------
def read_parquet_in_chunks(file_path, chunk_size):
    total_rows = len(pd.read_parquet(file_path))
    for start_idx in range(0, total_rows, chunk_size):
        end_idx = min(start_idx + chunk_size, total_rows)
        chunk = pd.read_parquet(file_path).iloc[start_idx:end_idx]
        yield chunk
        
def get_parameters_hash(args):
    param_str = f"{args.input}{args.text_column}{args.min_chars}{args.gpu_batch_size}{args.cpu_batch_size}{args.chunk_size}"
    return hashlib.md5(param_str.encode()).hexdigest()        
        

# ---------------- Main ----------------
def main():
    parser = argparse.ArgumentParser(description='Checkpointed Hybrid GPU+CPU BART for Parquet files')
    parser.add_argument('--input', '-i', required=True)
    parser.add_argument('--output', '-o', required=True)
    parser.add_argument('--text_column', '-c', default='trans_review')
    parser.add_argument('--min_chars', '-m', type=int, default=100)
    parser.add_argument('--gpu_batch_size', '-g', type=int, default=32)
    parser.add_argument('--cpu_batch_size', '-b', type=int, default=16)
    parser.add_argument('--chunk_size', '-s', type=int, default=5000)
    parser.add_argument('--cpu_workers', '-w', type=int, default=0)
    parser.add_argument('--clear_every', '-e', type=int, default=10)

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[ERROR] Input file {args.input} not found!")
        sys.exit(1)

    clear_memory()
    summarizer = HybridSummarizer(gpu_batch_size=args.gpu_batch_size, cpu_workers=args.cpu_workers)

    output_path = Path(args.output)
    params_hash = get_parameters_hash(args)
    checkpoint_dir = output_path.parent / f"{output_path.stem}_checkpoints_{params_hash[:8]}"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_file = checkpoint_dir / "progress.json"

    
    # Load progress if exists
    if checkpoint_file.exists():
        with open(checkpoint_file, 'r') as f:
            progress = json.load(f)
        if progress.get('params_hash') != params_hash:    
            print("[WARNING] Parameters changed! Starting fresh...")
            processed_chunks = []
            progress = {'processed_chunks': [], 'total_processed': 0, 'start_time': time.time(), 'params_hash': params_hash}
        else:
            processed_chunks = progress.get('processed_chunks', [])
            print(f"[DEBUG] Resuming from {len(processed_chunks)} processed chunks")
    else:
        processed_chunks = []
        progress = {'processed_chunks': [], 'total_processed': 0, 'start_time': time.time(), 'params_hash': params_hash}

    total_rows = len(pd.read_parquet(args.input))
    total_chunks = math.ceil(total_rows / args.chunk_size)
    print(f"[DEBUG] Total rows: {total_rows}, total chunks: {total_chunks}")

    chunk_iterator = read_parquet_in_chunks(args.input, args.chunk_size)
    total_processed = progress.get('total_processed', 0)
    start_time = progress.get('start_time', time.time())

    for chunk_idx, chunk in enumerate(tqdm(chunk_iterator, total=total_chunks, desc="Chunks")):
        if chunk_idx in processed_chunks:
            print(f"[DEBUG] Skipping chunk {chunk_idx} (already processed)")
            continue

        print(f"[DEBUG] Processing chunk {chunk_idx} ({len(chunk)} rows)")
        clear_memory()

        if args.text_column not in chunk.columns:
            print(f"[ERROR] Column {args.text_column} not found")
            sys.exit(1)

        chunk['cleaned_text'] = chunk[args.text_column].apply(lambda x: ' '.join(str(x).split()) if pd.notna(x) else "")
        chunk['char_count'] = chunk['cleaned_text'].str.len()

        summarize_mask = chunk['char_count'] > args.min_chars
        to_summarize = chunk[summarize_mask]
        keep_original = chunk[~summarize_mask]

        print(f"[DEBUG] Chunk {chunk_idx}: {len(to_summarize)} summaries to generate, {len(keep_original)} to keep")

        # ---- GPU processing (ALL texts when CPU workers = 0) ----
        if args.cpu_workers == 0:
            # Process ALL texts on GPU
            gpu_texts = to_summarize['cleaned_text'].tolist()
            gpu_indices = to_summarize.index.tolist()
        else:
            # Original split logic
            gpu_texts = to_summarize['cleaned_text'].tolist()[:len(to_summarize)//2]
            gpu_indices = to_summarize.index.tolist()[:len(to_summarize)//2]

        gpu_summaries = summarizer.summarize_gpu_batch(gpu_texts) if gpu_texts else []

        # ---- CPU processing (only if workers > 0) ----
        cpu_summaries_dict = {}
        if args.cpu_workers > 0:
            cpu_texts = to_summarize['cleaned_text'].tolist()[len(to_summarize)//2:]
            cpu_indices = to_summarize.index.tolist()[len(to_summarize)//2:]

        
            if cpu_texts:
                print(f"[DEBUG] Sending {len(cpu_texts)} texts to CPU pool with {args.cpu_workers} workers")
                cpu_batches = []
                for i in range(0, len(cpu_texts), args.cpu_batch_size):
                    batch_texts = cpu_texts[i:i + args.cpu_batch_size]
                    batch_indices = cpu_indices[i:i + args.cpu_batch_size]
                    cpu_batches.append((batch_texts, batch_indices))

                with ProcessPoolExecutor(max_workers=args.cpu_workers, initializer=init_cpu_worker) as executor:
                    futures = [executor.submit(process_cpu_batch, b) for b in cpu_batches]
                    batch_count = 0
                    for future in tqdm(as_completed(futures), total=len(futures), desc="CPU batches"):
                        batch_results = future.result()
                        for idx, summary in batch_results:
                            cpu_summaries_dict[idx] = summary
                        batch_count += 1
                        if batch_count % args.clear_every == 0:
                            clear_memory()

        # ---- Combine results ----
        summaries_dict = {idx: s for idx, s in zip(gpu_indices, gpu_summaries)}
        summaries_dict.update(cpu_summaries_dict)

        to_summarize = to_summarize.copy()
        to_summarize['summary'] = to_summarize.index.map(lambda x: summaries_dict.get(x, ""))
        to_summarize['was_summarized'] = to_summarize.index.map(lambda x: x in summaries_dict)

        keep_original = keep_original.copy()
        keep_original['summary'] = keep_original['cleaned_text']
        keep_original['was_summarized'] = False

        chunk_result = pd.concat([to_summarize, keep_original], ignore_index=True)

        # Save chunk immediately
        partial_output = checkpoint_dir / f"chunk_{chunk_idx}.parquet"
        chunk_result.to_parquet(partial_output)
        print(f"[DEBUG] Saved chunk {chunk_idx} to {partial_output}")

        # Update progress
        processed_chunks.append(chunk_idx)
        total_processed += len(chunk_result)
        progress.update({'processed_chunks': processed_chunks, 'total_processed': total_processed, 'last_update': time.time()})
        with open(checkpoint_file, 'w') as f:
            json.dump(progress, f, indent=2)

        del chunk, to_summarize, keep_original, chunk_result
        clear_memory()

    #  Combine all chunks at the end
    print("[DEBUG] Combining all chunks into final output...")
    final_chunks = list(checkpoint_dir.glob("chunk_*.parquet"))
    if not final_chunks:
        print("[ERROR] No chunk files found to combine!")
        sys.exit(1)

    try:
        final_df = pd.concat([pd.read_parquet(f) for f in final_chunks], ignore_index=True)
        
        # Atomic write
        temp_output = output_path.with_suffix('.tmp')
        final_df.to_parquet(temp_output, index=False)
        temp_output.rename(output_path)  # Atomic rename
        
        print(f"[DEBUG] Saved final output to {args.output}")
        
        # Clean up only after successful write
        for f in final_chunks:
            f.unlink()
        if checkpoint_file.exists():
            checkpoint_file.unlink()
        checkpoint_dir.rmdir()
        
    except Exception as e:
        print(f"[ERROR] Failed to combine chunks: {e}")
        print("[INFO] Chunk files preserved for recovery")
        sys.exit(1)

    total_time = time.time() - start_time
    summarized = final_df['was_summarized'].sum()
    print(f"[DEBUG] Processing complete! Total reviews: {len(final_df)}, Summarized: {summarized}")
    print(f"[DEBUG] Total time: {total_time/3600:.2f} hours, Avg rate: {len(final_df)/total_time:.1f} reviews/sec")

if __name__ == "__main__":
    mp.freeze_support()
    main()
