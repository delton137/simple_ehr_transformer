#!/usr/bin/env python3
"""
Post-process a processed dataset to shrink the vocabulary to MAX_VOCAB_SIZE by
dropping lowest-frequency concept tokens and remapping timelines.

- Preserves special tokens (<PAD>, <UNK>, <EOS>, <SOS>) and structural tokens
  (EVENT_*, AGE_*, TIME_*, Q*, GENDER_*, RACE_*, YEAR_*, VISIT_TYPE_*, UNIT_*).
- Prunes only clinical concept tokens: CONDITION_*, DRUG_*, PROCEDURE_*,
  MEASUREMENT_*, OBSERVATION_* by lowest frequency first.

Outputs a new directory with remapped vocabulary and timelines.
"""

import os
import argparse
import pickle
from collections import Counter
from typing import Dict, List, Tuple
from tqdm import tqdm

# Default maximum vocabulary size
MAX_VOCAB_SIZE = 30000


def load_pickle(path: str):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_pickle(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def compute_token_frequencies(tokenized_timelines: Dict[int, List[int]]) -> Counter:
    freq = Counter()
    for tokens in tokenized_timelines.values():
        freq.update(tokens)
    return freq


def build_keep_sets(vocab: Dict[str, int]) -> Tuple[Dict[str, int], set, set]:
    """Return (id_to_token, always_keep_token_strings, prunable_token_strings)."""
    id_to_token = {tid: tok for tok, tid in vocab.items()}

    # Always keep: special, structural, demographics, units, visit type
    always_keep_prefixes = (
        'EVENT_', 'AGE_', 'TIME_', 'Q', 'GENDER_', 'RACE_', 'YEAR_', 'VISIT_TYPE_', 'UNIT_'
    )

    # Prune only clinical concept tokens
    prunable_prefixes = (
        'CONDITION_', 'DRUG_', 'PROCEDURE_', 'MEASUREMENT_', 'OBSERVATION_'
    )

    always_keep = set()
    prunable = set()

    # Special tokens by string
    for sp in ('<PAD>', '<UNK>', '<EOS>', '<SOS>'):
        if sp in vocab:
            always_keep.add(sp)

    for tok in vocab.keys():
        if tok in always_keep:
            continue
        if tok.startswith(prunable_prefixes):
            prunable.add(tok)
        elif tok.startswith(always_keep_prefixes):
            always_keep.add(tok)
        else:
            # Unknown category: keep by default
            always_keep.add(tok)

    return id_to_token, always_keep, prunable


def shrink_vocab(data_dir: str, output_dir: str, max_vocab_size: int) -> None:
    vocab_path = os.path.join(data_dir, 'vocabulary.pkl')
    timelines_path = os.path.join(data_dir, 'tokenized_timelines.pkl')

    if not os.path.exists(vocab_path) or not os.path.exists(timelines_path):
        raise FileNotFoundError('vocabulary.pkl or tokenized_timelines.pkl not found in data_dir')

    vocab: Dict[str, int] = load_pickle(vocab_path)
    tokenized_timelines: Dict[int, List[int]] = load_pickle(timelines_path)

    id_to_token, always_keep, prunable = build_keep_sets(vocab)

    # Compute frequencies by token string
    freqs_by_id = compute_token_frequencies(tokenized_timelines)
    freqs_by_token = Counter({id_to_token.get(i, '<UNK>'): c for i, c in freqs_by_id.items()})

    current_size = len(vocab)
    print(f"Original vocabulary size: {current_size}")
    print(f"Always-keep tokens: {len(always_keep)} | Prunable concept tokens: {len(prunable)}")

    # Determine capacity for prunable tokens
    capacity = max_vocab_size - len(always_keep)
    if capacity <= 0:
        raise ValueError(f"MAX_VOCAB_SIZE={max_vocab_size} too small; needs > always_keep={len(always_keep)}")

    # Rank prunable tokens by frequency (desc), then by token string for stability
    prunable_sorted = sorted(
        prunable,
        key=lambda t: (-freqs_by_token.get(t, 0), t)
    )
    kept_prunable = prunable_sorted[:capacity]

    # Build new vocabulary: deterministic order
    new_vocab: Dict[str, int] = {}
    def add(tok: str):
        if tok not in new_vocab:
            new_vocab[tok] = len(new_vocab)

    # 1) Special tokens in canonical order
    for sp in ('<PAD>', '<UNK>', '<EOS>', '<SOS>'):
        if sp in vocab:
            add(sp)

    # 2) Structural/demographic units/visit/event/age/time/quantile/etc.
    structural_prefix_order = [
        'EVENT_', 'AGE_', 'TIME_', 'Q', 'GENDER_', 'RACE_', 'YEAR_', 'VISIT_TYPE_', 'UNIT_'
    ]
    for pref in structural_prefix_order:
        for tok in sorted(always_keep):
            if tok.startswith(pref):
                add(tok)

    # 3) Any remaining always_keep (non-prunable) tokens
    for tok in sorted(always_keep):
        add(tok)

    # 4) Top prunable concept tokens by frequency
    for tok in kept_prunable:
        add(tok)

    print(f"New vocabulary size: {len(new_vocab)} (target max {max_vocab_size})")

    # Build id remap (old_id -> new_id), unknowns -> <UNK>
    unk_id = new_vocab.get('<UNK>', 0)
    oldid_to_newid: Dict[int, int] = {}
    for tok, old_id in vocab.items():
        new_id = new_vocab.get(tok, unk_id)
        oldid_to_newid[old_id] = new_id

    # Remap timelines
    remapped: Dict[int, List[int]] = {}
    replaced = 0
    total = 0
    for pid, seq in tqdm(tokenized_timelines.items(), desc='Remapping timelines', unit='pt'):
        new_seq = []
        for tid in seq:
            total += 1
            nid = oldid_to_newid.get(tid, unk_id)
            if nid == unk_id and tid != unk_id:
                replaced += 1
            new_seq.append(nid)
        remapped[pid] = new_seq

    print(f"Tokens remapped to <UNK>: {replaced} / {total} ({(replaced/total*100 if total else 0):.2f}%)")

    # Write outputs
    out_dir = output_dir or f"{data_dir}_v{max_vocab_size}"
    os.makedirs(out_dir, exist_ok=True)
    save_pickle(new_vocab, os.path.join(out_dir, 'vocabulary.pkl'))
    save_pickle(remapped, os.path.join(out_dir, 'tokenized_timelines.pkl'))
    # Save mapping for reference
    save_pickle(oldid_to_newid, os.path.join(out_dir, 'oldid_to_newid.pkl'))

    print(f"Saved pruned dataset to: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description='Shrink vocabulary by pruning low-frequency concept tokens')
    parser.add_argument('--data_dir', required=True, help='Input processed data directory (contains vocabulary.pkl, tokenized_timelines.pkl)')
    parser.add_argument('--output_dir', default=None, help='Output directory (default: <data_dir>_v<MAX_VOCAB_SIZE>)')
    parser.add_argument('--max_vocab_size', type=int, default=MAX_VOCAB_SIZE, help='Maximum vocabulary size')
    args = parser.parse_args()

    shrink_vocab(args.data_dir, args.output_dir, args.max_vocab_size)


if __name__ == '__main__':
    main()


