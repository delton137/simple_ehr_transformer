#!/usr/bin/env python3
"""
Export tokenizer vocabulary and optional sample timelines
- Loads processed_data/{vocabulary.pkl, tokenized_timelines.pkl}
- Groups tokens by category (EVENT_, AGE_, TIME_, Q, GENDER_, RACE_, YEAR_, VISIT_TYPE_, UNIT_, CONDITION_, DRUG_, PROCEDURE_, MEASUREMENT_, OBSERVATION_)
- Writes a human-readable report (txt/markdown) and optional TSV
- Optionally emits decoded example timelines (>100 tokens preferred)
"""

import os
import argparse
import pickle
from collections import defaultdict, Counter
from typing import Dict, List, Tuple

CATEGORIES = [
    'EVENT_', 'AGE_', 'TIME_', 'Q', 'GENDER_', 'RACE_', 'YEAR_', 'VISIT_TYPE_', 'UNIT_',
    'CONDITION_', 'DRUG_', 'PROCEDURE_', 'MEASUREMENT_', 'OBSERVATION_'
]


def load_processed(data_dir: str):
    with open(os.path.join(data_dir, 'vocabulary.pkl'), 'rb') as f:
        vocab = pickle.load(f)
    timelines = None
    pt_path = os.path.join(data_dir, 'patient_timelines.pkl')
    if os.path.exists(pt_path):
        with open(pt_path, 'rb') as f:
            timelines = pickle.load(f)
    else:
        # Fall back to tokenized timelines if patient_timelines not present
        tt_path = os.path.join(data_dir, 'tokenized_timelines.pkl')
        if os.path.exists(tt_path):
            with open(tt_path, 'rb') as f:
                timelines = pickle.load(f)
    return vocab, timelines


def group_vocabulary(vocab: Dict[str, int]) -> Dict[str, List[Tuple[str, int]]]:
    grouped = defaultdict(list)
    for token, tok_id in vocab.items():
        matched = False
        for cat in CATEGORIES:
            if token.startswith(cat):
                grouped[cat].append((token, tok_id))
                matched = True
                break
        if not matched:
            grouped['OTHER'].append((token, tok_id))
    # sort by token_id within each group
    for k in grouped:
        grouped[k].sort(key=lambda x: x[1])
    return grouped


essential_order = [
    'EVENT_', 'AGE_', 'TIME_', 'Q', 'GENDER_', 'RACE_', 'YEAR_', 'VISIT_TYPE_', 'UNIT_',
    'CONDITION_', 'DRUG_', 'PROCEDURE_', 'MEASUREMENT_', 'OBSERVATION_', 'OTHER'
]


def count_token_freqs(timelines: Dict[int, List[int]], vocab: Dict[str, int]) -> Dict[str, int]:
    if timelines is None:
        return {}
    freqs = Counter()
    for _, seq in timelines.items():
        freqs.update(seq)
    # map id->count
    id_to_count = dict(freqs)
    # map token->count
    return {tok: id_to_count.get(tok_id, 0) for tok, tok_id in vocab.items()}


def write_report_txt(path: str, grouped: Dict[str, List[Tuple[str, int]]], counts: Dict[str, int]):
    with open(path, 'w') as f:
        total = sum(len(v) for v in grouped.values())
        f.write(f"Vocabulary report\n")
        f.write(f"Total tokens: {total}\n\n")
        for cat in essential_order:
            if cat not in grouped:
                continue
            f.write(f"[{cat}]\n")
            for token, tok_id in grouped[cat]:
                cnt = counts.get(token, 0)
                f.write(f"{tok_id}\t{token}\t{cnt}\n")
            f.write("\n")


def write_report_tsv(path: str, grouped: Dict[str, List[Tuple[str, int]]], counts: Dict[str, int]):
    with open(path, 'w') as f:
        f.write("category\ttoken_id\ttoken\tcount\n")
        for cat in essential_order:
            if cat not in grouped:
                continue
            for token, tok_id in grouped[cat]:
                cnt = counts.get(token, 0)
                f.write(f"{cat}\t{tok_id}\t{token}\t{cnt}\n")


def write_examples(path: str, timelines: Dict[int, List[int]], vocab: Dict[str, int], min_len: int = 100, max_tokens_preview: int = 400):
    if timelines is None:
        return
    id_to_token = {v: k for k, v in vocab.items()}
    # find candidates > min_len, else pick the longest two
    items = list(timelines.items())
    long_items = [(pid, seq) for pid, seq in items if len(seq) > min_len]
    if not long_items:
        long_items = sorted(items, key=lambda x: len(x[1]), reverse=True)[:2]
    else:
        # pick up to two examples
        long_items = long_items[:2]
    with open(path, 'w') as f:
        f.write("Supplementary: Example Tokenized Timelines (decoded tokens)\n\n")
        for i, (pid, seq) in enumerate(long_items, start=1):
            f.write(f"Patient Health Timeline {i}\n")
            f.write(f"Patient ID: {pid}\n")
            f.write(f"Sequence length: {len(seq)}\n")
            decoded = [id_to_token.get(t, f'UNKNOWN_{t}') for t in seq]
            preview = decoded[:max_tokens_preview]
            # wrap at 8 per line for readability
            for j in range(0, len(preview), 8):
                f.write(", ".join(preview[j:j+8]) + "\n")
            f.write("\n")


def main():
    ap = argparse.ArgumentParser(description='Export tokenizer vocabulary and sample timelines')
    ap.add_argument('--data_dir', type=str, default='processed_data', help='Processed data directory')
    ap.add_argument('--tag', type=str, default=None, help='Dataset tag (sets data_dir=processed_data_{tag})')
    ap.add_argument('--out_txt', type=str, default='vocabulary_report.txt', help='Path to write text report')
    ap.add_argument('--out_tsv', type=str, default='vocabulary_report.tsv', help='Path to write TSV report')
    ap.add_argument('--examples', type=str, default='vocabulary_examples.txt', help='Path to write decoded examples')
    ap.add_argument('--include_examples', action='store_true', help='Also write decoded sample timelines (>100 tokens preferred)')
    args = ap.parse_args()

    if args.tag and not args.data_dir.startswith('processed_data_'):
        args.data_dir = f"processed_data_{args.tag}"
    if not os.path.exists(args.data_dir):
        raise FileNotFoundError(f"Data dir not found: {args.data_dir}")

    vocab, timelines = load_processed(args.data_dir)
    grouped = group_vocabulary(vocab)
    counts = count_token_freqs(timelines, vocab)

    write_report_txt(args.out_txt, grouped, counts)
    write_report_tsv(args.out_tsv, grouped, counts)
    if args.include_examples and timelines:
        write_examples(args.examples, timelines, vocab)

    print(f"✅ Wrote vocabulary reports to:\n  - {args.out_txt}\n  - {args.out_tsv}")
    if args.include_examples and timelines:
        print(f"✅ Wrote examples to: {args.examples}")


if __name__ == '__main__':
    main()
