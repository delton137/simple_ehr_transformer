#!/usr/bin/env python3
"""
Build random-forest-ready feature matrices (X, y) from processed OMOP tokenization outputs.

Inputs (from a processed data directory):
- tokenization.yaml: contains 'concepts' listing with token metadata and counts, and 'vocabulary' (ordered)
- vocabulary.pkl: dict[str token_name -> int token_id]
- tokenized_timelines.pkl: dict[int patient_id -> list[int token_id]]

Behavior:
- Reads tokenization.yaml and builds the feature list from 'concepts', ordered by 'count' desc
- For each patient, counts occurrences of each feature token (by token_id) → X (num_patients x num_features)
- Builds y as a binary vector indicating presence of a specified target concept token in the patient's timeline
- Optionally excludes the target token from the features (to avoid leakage)

Outputs:
- X.npy (int32), y.npy (int8), patient_ids.npy (int64)
- features.tsv: columns [index, token, concept_id, count]
"""

import os
import argparse
import pickle
import yaml
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter
from tqdm import tqdm
try:
    import scipy.sparse as sp
except Exception:
    sp = None


CONCEPT_PREFIXES = (
    "CONDITION_", "CONDITION_OCCURRENCE_",
    "DRUG_", "DRUG_EXPOSURE_",
    "PROCEDURE_", "PROCEDURE_OCCURRENCE_",
    "MEASUREMENT_", "OBSERVATION_",
)


def load_processed(data_dir: str):
    vocab_path = os.path.join(data_dir, 'vocabulary.pkl')
    tt_path = os.path.join(data_dir, 'tokenized_timelines.pkl')
    spec_path = os.path.join(data_dir, 'tokenization.yaml')
    if not (os.path.exists(vocab_path) and os.path.exists(tt_path) and os.path.exists(spec_path)):
        raise FileNotFoundError(f"Missing required files in {data_dir}. Need vocabulary.pkl, tokenized_timelines.pkl, tokenization.yaml")
    
    print("Loading vocabulary...")
    with open(vocab_path, 'rb') as f:
        vocab: Dict[str, int] = pickle.load(f)
    
    print("Loading tokenized timelines...")
    with open(tt_path, 'rb') as f:
        tokenized_timelines: Dict[int, List[int]] = pickle.load(f)
    
    print("Loading tokenization specification...")
    with open(spec_path, 'r') as f:
        spec = yaml.safe_load(f)
    
    return vocab, tokenized_timelines, spec


def feature_tokens_from_spec(spec: dict, min_count: int = 0) -> List[Tuple[str, int]]:
    """Return list of (token_name, count), ordered by count desc, optionally filtered.
    Expects spec['concepts'] entries with 'token' and 'count'. Falls back to empty list if missing.
    """
    concepts = spec.get('concepts', []) or []
    feats = []
    for entry in concepts:
        tok = entry.get('token')
        cnt = int(entry.get('count', 0))
        if tok is None:
            continue
        if cnt < min_count:
            continue
        feats.append((tok, cnt))
    # Sort by count desc, then token name for stability
    feats.sort(key=lambda x: (-x[1], x[0]))
    return feats


def expand_variants_from_concept_id(spec: dict, concept_id: int) -> List[str]:
    """Find all token names in spec['concepts'] matching a concept_id across prefixes."""
    out: List[str] = []
    for entry in spec.get('concepts', []) or []:
        try:
            cid = int(entry.get('concept_id'))
        except Exception:
            continue
        if cid == concept_id:
            tok = entry.get('token')
            if tok:
                out.append(tok)
    return out


def main():
    ap = argparse.ArgumentParser(description='Build RF feature matrix (X, y) from processed tokenization outputs')
    ap.add_argument('--data_dir', required=True, help='Processed data directory (with tokenization.yaml, vocabulary.pkl, tokenized_timelines.pkl)')
    ap.add_argument('--out_dir', default=None, help='Output directory (default: <data_dir>_rf)')
    ap.add_argument('--min_count', type=int, default=0, help='Drop tokens with total count < min_count (from spec)')
    ap.add_argument('--sparse', action='store_true', help='Build a sparse CSR matrix (saves X_sparse.npz) to reduce RAM')
    ap.add_argument('--target_concept_id', type=int, default=None, help='Target concept_id; all tokens in spec with this concept_id will be removed and added to y')
    args = ap.parse_args()

    out_dir = args.out_dir or f"{args.data_dir.rstrip(os.sep)}_rf"
    os.makedirs(out_dir, exist_ok=True)

    vocab, tokenized_timelines, spec = load_processed(args.data_dir)

    # Derive target token name(s)
    target_tokens = expand_variants_from_concept_id(spec, int(args.target_concept_id))

    # Build feature set from spec concepts ordered by count desc
    print("Building feature set from concepts...")
    feats_with_counts = feature_tokens_from_spec(spec, min_count=args.min_count)
    feature_tokens = [t for t, _ in feats_with_counts]
    feature_counts = [c for _, c in feats_with_counts]
    print(f"Found {len(feature_tokens)} features with min_count >= {args.min_count}")

    # Always remove target tokens from features to avoid leakage
    print(f"Target tokens: {target_tokens}")
    feature_mask = [t not in set(target_tokens) for t in feature_tokens]
    feature_tokens = [t for t, keep in zip(feature_tokens, feature_mask) if keep]
    feature_counts = [c for c, keep in zip(feature_counts, feature_mask) if keep]
    print(f"Features after removing targets: {len(feature_tokens)}")

    # Map feature tokens to ids via vocabulary
    token_name_to_id = vocab  # str -> int
    feature_ids: List[int] = []
    feature_rows = []
    print("Mapping feature tokens to IDs...")
    for idx, (tok, cnt) in tqdm(enumerate(zip(feature_tokens, feature_counts)), 
                                total=len(feature_tokens), 
                                desc="Processing features"):
        tid = token_name_to_id.get(tok)
        if tid is None:
            # Skip tokens not in vocab (mismatch); log row but with -1 id
            feature_rows.append((idx, tok, -1, cnt))
            continue
        feature_ids.append(tid)
        # Find concept_id for this token from spec
        concept_id = None
        for entry in spec.get('concepts', []) or []:
            if entry.get('token') == tok:
                concept_id = entry.get('concept_id')
                break
        feature_rows.append((idx, tok, concept_id if concept_id is not None else '', cnt))

    # Target id set (by name(s))
    target_id_set = {token_name_to_id[tok] for tok in target_tokens if tok in token_name_to_id}
    if not target_id_set:
        raise ValueError('Resolved target token(s) are not present in vocabulary')

    # Build X and y
    patient_ids = sorted(tokenized_timelines.keys())
    num_patients = len(patient_ids)
    num_features = len(feature_tokens)
    y = np.zeros((num_patients,), dtype=np.uint8)

    feature_id_to_col = {tid: col for col, tid in enumerate(feature_ids) if tid != -1}

    use_sparse = bool(args.sparse)
    # Auto-enable sparse if very wide
    if not use_sparse and num_features > 10000:
        use_sparse = True

    print(f"Building feature matrix for {num_patients} patients... (sparse={use_sparse})")
    if use_sparse:
        if sp is None:
            raise RuntimeError("scipy is required for --sparse output (pip install scipy)")
        rows: List[int] = []
        cols: List[int] = []
        data: List[int] = []
        for row, pid in tqdm(enumerate(patient_ids), total=num_patients, desc="Processing patients"):
            seq = tokenized_timelines[pid]
            counts = Counter(seq)
            # Emit only non-zero features
            for tid, cnt in counts.items():
                col = feature_id_to_col.get(tid)
                if col is not None and cnt:
                    rows.append(row)
                    cols.append(col)
                    data.append(int(cnt))
            # Target label: presence of any target id
            y[row] = 1 if any((tid in counts) for tid in target_id_set) else 0
        X_sparse = sp.coo_matrix((data, (rows, cols)), shape=(num_patients, num_features), dtype=np.int32).tocsr()
    else:
        X = np.zeros((num_patients, num_features), dtype=np.int32)
        for row, pid in tqdm(enumerate(patient_ids), total=num_patients, desc="Processing patients"):
            seq = tokenized_timelines[pid]
            counts = Counter(seq)
            # Features
            for tid, col in feature_id_to_col.items():
                X[row, col] = counts.get(tid, 0)
            # Target label: presence of any target id
            y[row] = 1 if any((tid in counts) for tid in target_id_set) else 0

    # Save outputs
    print("Saving outputs...")
    if use_sparse:
        sp.save_npz(os.path.join(out_dir, 'X_sparse.npz'), X_sparse)
    else:
        np.save(os.path.join(out_dir, 'X.npy'), X)
    np.save(os.path.join(out_dir, 'y.npy'), y)
    np.save(os.path.join(out_dir, 'patient_ids.npy'), np.array(patient_ids, dtype=np.int64))
    
    print("Writing features.tsv...")
    with open(os.path.join(out_dir, 'features.tsv'), 'w') as f:
        f.write("index\ttoken\tconcept_id\tcount\n")
        for idx, (tok, concept_id, cnt) in tqdm(zip(feature_tokens, [r[2] for r in feature_rows], feature_counts), 
                                                 total=len(feature_tokens), 
                                                 desc="Writing features"):
            f.write(f"{idx}\t{tok}\t{concept_id}\t{cnt}\n")

    print(f"Saved X, y, patient_ids, and features to {out_dir}")
    print(f"Final dataset: {num_patients} patients × {num_features} features")
    print(f"Target distribution: {np.sum(y)} positive, {num_patients - np.sum(y)} negative")
    print(f"Feature matrix shape: X={X.shape}, y={y.shape}")


if __name__ == '__main__':
    main()


