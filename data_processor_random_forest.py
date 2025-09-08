#!/usr/bin/env python3
"""
Build random-forest-ready feature matrices (X, y) with temporal split.

Inputs (from a processed data directory):
- tokenization.yaml: contains 'concepts' listing with token metadata and counts, and 'vocabulary' (ordered)
- vocabulary.pkl: dict[str token_name -> int token_id]
- patient_timelines.pkl: dict[int patient_id -> list[dict events with 'timestamp' and concept ids]]

Temporal behavior:
- For each patient, determine start_time as the earliest non-static event timestamp
- First window (train features X): [start_time, start_time + train_days)
- Second window (labels y): [start_time + train_days, start_time + train_days + test_days)
- X counts concept tokens occurring in the first window only
- y=1 if any target concept token occurs in the second window; otherwise 0
- Target tokens are excluded from X to avoid leakage

Outputs:
- For train data_dir: X.npy (int32) or X_sparse.npz when --sparse, y.npy (uint8), patient_ids.npy (int64)
- For optional test_data_dir: X.npy/X_sparse.npz, y.npy, patient_ids.npy using the SAME tokenization as train
- features.tsv (in train out dir): columns [index, token, concept_id, count]
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
from datetime import datetime, timedelta
import pandas as pd


CONCEPT_PREFIXES = (
    "CONDITION_", "CONDITION_OCCURRENCE_",
    "DRUG_", "DRUG_EXPOSURE_",
    "PROCEDURE_", "PROCEDURE_OCCURRENCE_",
    "MEASUREMENT_", "OBSERVATION_",
)


def load_processed(data_dir: str):
    vocab_path = os.path.join(data_dir, 'vocabulary.pkl')
    spec_path = os.path.join(data_dir, 'tokenization.yaml')
    pt_path = os.path.join(data_dir, 'patient_timelines.pkl')
    missing = []
    if not os.path.exists(vocab_path):
        missing.append('vocabulary.pkl')
    if not os.path.exists(spec_path):
        missing.append('tokenization.yaml')
    if not os.path.exists(pt_path):
        missing.append('patient_timelines.pkl')
    if missing:
        raise FileNotFoundError(f"Missing required files in {data_dir}: {', '.join(missing)}")

    print("Loading vocabulary...")
    with open(vocab_path, 'rb') as f:
        vocab: Dict[str, int] = pickle.load(f)

    print("Loading patient timelines...")
    with open(pt_path, 'rb') as f:
        patient_timelines: Dict[int, List[Dict]] = pickle.load(f)

    print("Loading tokenization specification...")
    with open(spec_path, 'r') as f:
        spec = yaml.safe_load(f)

    return vocab, patient_timelines, spec


def build_timelines_from_events_partitioned(data_dir: str) -> Dict[int, List[dict]]:
    """Reconstruct minimal timelines from events_partitioned parquet files.
    Requires columns: person_id, ts, et, cid, value_as_number, unit_concept_id
    """
    base = os.path.join(data_dir, 'events_partitioned')
    if not os.path.isdir(base):
        raise FileNotFoundError(f"events_partitioned directory not found in {data_dir}")
    from pathlib import Path
    files = list(Path(base).rglob('*.parquet'))
    timelines: Dict[int, List[dict]] = {}
    for fp in tqdm(files, desc='Reconstructing timelines from events_partitioned', unit='file'):
        try:
            df = pd.read_parquet(str(fp))
        except Exception:
            continue
        if df is None or len(df) == 0:
            continue
        required = [c for c in ['person_id','ts','et','cid','value_as_number','unit_concept_id'] if c in df.columns]
        if not {'person_id','ts','et','cid'}.issubset(set(required)):
            continue
        df = df.sort_values(['person_id','ts'])
        for pid, sub in df.groupby('person_id'):
            pid = int(pid)
            seq = timelines.get(pid, [])
            for _, row in sub.iterrows():
                ts = row.get('ts')
                et = row.get('et')
                cid = row.get('cid')
                if ts is None or pd.isna(ts) or et is None or cid is None:
                    continue
                ev: Dict[str, object] = {'timestamp': ts, 'event_type': et}
                if et == 'condition':
                    ev['condition_concept_id'] = int(cid)
                elif et == 'medication':
                    ev['drug_concept_id'] = int(cid)
                elif et == 'procedure':
                    ev['procedure_concept_id'] = int(cid)
                elif et == 'measurement':
                    ev['measurement_concept_id'] = int(cid)
                    if 'value_as_number' in df.columns:
                        ev['value_as_number'] = row.get('value_as_number')
                    if 'unit_concept_id' in df.columns:
                        ev['unit_concept_id'] = row.get('unit_concept_id')
                elif et == 'observation':
                    ev['observation_concept_id'] = int(cid)
                seq.append(ev)
            if seq:
                # ensure sorted
                seq.sort(key=lambda e: e.get('timestamp'))
                timelines[pid] = seq
    return timelines


def load_patient_timelines_only(data_dir: str, allow_events_fallback: bool = True) -> Dict[int, List[dict]]:
    """Load patient_timelines.pkl; optionally fallback to events_partitioned if empty/missing."""
    pt_path = os.path.join(data_dir, 'patient_timelines.pkl')
    timelines: Optional[Dict[int, List[dict]]] = None
    if os.path.exists(pt_path):
        try:
            print(f"Loading patient timelines from {data_dir}...")
            with open(pt_path, 'rb') as f:
                timelines = pickle.load(f)
            if not isinstance(timelines, dict) or len(timelines) == 0:
                timelines = None
        except Exception:
            timelines = None
    if timelines is None and allow_events_fallback:
        print("patient_timelines.pkl missing or empty; falling back to events_partitioned reconstruction...")
        timelines = build_timelines_from_events_partitioned(data_dir)
    if timelines is None:
        raise FileNotFoundError(f"Could not load timelines from {data_dir}")
    return timelines


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


def event_to_token_name(event: Dict) -> Optional[str]:
    """Map a patient timeline event to a concept token name used in vocabulary.
    Only concept-bearing events are considered.
    """
    et = event.get('event_type')
    if et == 'condition':
        cid = event.get('condition_concept_id')
        if cid is not None:
            return f"CONDITION_OCCURRENCE_{int(cid)}"
    elif et == 'medication':
        cid = event.get('drug_concept_id')
        if cid is not None:
            return f"DRUG_EXPOSURE_{int(cid)}"
    elif et == 'procedure':
        cid = event.get('procedure_concept_id')
        if cid is not None:
            return f"PROCEDURE_OCCURRENCE_{int(cid)}"
    elif et == 'measurement':
        cid = event.get('measurement_concept_id')
        if cid is not None:
            return f"MEASUREMENT_{int(cid)}"
    elif et == 'observation':
        cid = event.get('observation_concept_id')
        if cid is not None:
            return f"OBSERVATION_{int(cid)}"
    return None


def find_start_time(timeline: List[Dict]) -> Optional[datetime]:
    """Earliest timestamp among non-static events; returns None if not found."""
    start: Optional[datetime] = None
    for ev in timeline:
        if ev.get('event_type') in { 'condition','medication','procedure','measurement','observation','admission','discharge','death' }:
            ts = ev.get('timestamp')
            if isinstance(ts, datetime):
                if start is None or ts < start:
                    start = ts
    return start


def main():
    ap = argparse.ArgumentParser(description='Build RF feature matrix (X, y) from processed tokenization outputs')
    ap.add_argument('--data_dir', required=True, help='Processed data directory (with tokenization.yaml, vocabulary.pkl, tokenized_timelines.pkl)')
    ap.add_argument('--out_dir', default=None, help='Output directory (default: <data_dir>_rf)')
    ap.add_argument('--min_count', type=int, default=0, help='Drop tokens with total count < min_count (from spec)')
    ap.add_argument('--sparse', action='store_true', help='Build a sparse CSR matrix (saves X_sparse.npz) to reduce RAM')
    ap.add_argument('--target_concept_id', type=int, default=None, help='Target concept_id; all tokens in spec with this concept_id will be removed and added to y')
    ap.add_argument('--train_days', type=int, default=365, help='Length of first window in days for features (default: 365)')
    ap.add_argument('--test_days', type=int, default=365, help='Length of second window in days for labels (default: 365)')
    ap.add_argument('--test_data_dir', type=str, default=None, help='Optional processed test data directory (uses train tokenization)')
    ap.add_argument('--test_out_dir', type=str, default=None, help='Optional output dir for test matrices (default: <test_data_dir>_rf)')
    args = ap.parse_args()

    out_dir = args.out_dir or f"{args.data_dir.rstrip(os.sep)}_rf"
    os.makedirs(out_dir, exist_ok=True)

    vocab, patient_timelines, spec = load_processed(args.data_dir)

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

    def build_matrix_for_timelines(pt: Dict[int, List[dict]],
                                   vocab_map: Dict[str, int],
                                   feature_id_to_col: Dict[int, int],
                                   target_id_set_local: set,
                                   train_days: int,
                                   test_days: int,
                                   want_sparse: bool):
        """Build X/y given timelines and a fixed feature mapping."""
        patient_ids_local = sorted(pt.keys())
        n_pat = len(patient_ids_local)
        n_feat = len(feature_id_to_col)
        y_local = np.zeros((n_pat,), dtype=np.uint8)
        use_sparse_local = bool(want_sparse)
        if not use_sparse_local and n_feat > 10000:
            use_sparse_local = True
        train_delta_local = timedelta(days=int(train_days))
        test_delta_local = timedelta(days=int(test_days))
        if use_sparse_local:
            if sp is None:
                raise RuntimeError("scipy is required for --sparse output (pip install scipy)")
            rows: List[int] = []
            cols: List[int] = []
            data: List[int] = []
            for row, pid in tqdm(enumerate(patient_ids_local), total=n_pat, desc="Processing patients"):
                timeline = pt.get(pid, [])
                start = find_start_time(timeline)
                if start is None:
                    y_local[row] = 0
                    continue
                train_end = start + train_delta_local
                test_end = train_end + test_delta_local
                counts: Dict[int, int] = {}
                for ev in timeline:
                    ts = ev.get('timestamp')
                    if not isinstance(ts, datetime):
                        continue
                    if ts < start or ts >= train_end:
                        continue
                    tok = event_to_token_name(ev)
                    if tok is None:
                        continue
                    tid = vocab_map.get(tok)
                    if tid is None:
                        continue
                    col = feature_id_to_col.get(tid)
                    if col is None:
                        continue
                    counts[col] = counts.get(col, 0) + 1
                for col, cnt in counts.items():
                    rows.append(row)
                    cols.append(col)
                    data.append(int(cnt))
                label = 0
                for ev in timeline:
                    ts = ev.get('timestamp')
                    if not isinstance(ts, datetime):
                        continue
                    if ts < train_end or ts >= test_end:
                        continue
                    tok = event_to_token_name(ev)
                    if tok is None:
                        continue
                    tid = vocab_map.get(tok)
                    if tid in target_id_set_local:
                        label = 1
                        break
                y_local[row] = label
            X_local = sp.coo_matrix((data, (rows, cols)), shape=(n_pat, len(feature_tokens)), dtype=np.int32).tocsr()
        else:
            X_local = np.zeros((n_pat, len(feature_tokens)), dtype=np.int32)
            for row, pid in tqdm(enumerate(patient_ids_local), total=n_pat, desc="Processing patients"):
                timeline = pt.get(pid, [])
                start = find_start_time(timeline)
                if start is None:
                    y_local[row] = 0
                    continue
                train_end = start + train_delta_local
                test_end = train_end + test_delta_local
                for ev in timeline:
                    ts = ev.get('timestamp')
                    if not isinstance(ts, datetime):
                        continue
                    if ts < start or ts >= train_end:
                        continue
                    tok = event_to_token_name(ev)
                    if tok is None:
                        continue
                    tid = vocab_map.get(tok)
                    if tid is None:
                        continue
                    col = feature_id_to_col.get(tid)
                    if col is None:
                        continue
                    X_local[row, col] += 1
                label = 0
                for ev in timeline:
                    ts = ev.get('timestamp')
                    if not isinstance(ts, datetime):
                        continue
                    if ts < train_end or ts >= test_end:
                        continue
                    tok = event_to_token_name(ev)
                    if tok is None:
                        continue
                    tid = vocab_map.get(tok)
                    if tid in target_id_set_local:
                        label = 1
                        break
                y_local[row] = label
        return X_local, y_local, np.array(patient_ids_local, dtype=np.int64), use_sparse_local

    # Build mapping structures
    feature_id_to_col = {tid: col for col, tid in enumerate(feature_ids) if tid != -1}
    use_sparse = bool(args.sparse)
    if not use_sparse and len(feature_tokens) > 10000:
        use_sparse = True
    print(f"Building TRAIN feature matrix for {len(patient_timelines)} patients with temporal split (sparse={use_sparse})")
    X_train, y_train, pids_train, used_sparse_train = build_matrix_for_timelines(
        patient_timelines, vocab, feature_id_to_col, target_id_set, args.train_days, args.test_days, use_sparse
    )

    # Save outputs
    print("Saving TRAIN outputs...")
    if used_sparse_train:
        sp.save_npz(os.path.join(out_dir, 'X_sparse.npz'), X_train)
    else:
        np.save(os.path.join(out_dir, 'X.npy'), X_train)
    np.save(os.path.join(out_dir, 'y.npy'), y_train)
    np.save(os.path.join(out_dir, 'patient_ids.npy'), pids_train)
    
    print("Writing features.tsv...")
    with open(os.path.join(out_dir, 'features.tsv'), 'w') as f:
        f.write("index\ttoken\tconcept_id\tcount\n")
        concept_ids = [r[2] for r in feature_rows]
        for idx in tqdm(range(len(feature_tokens)), total=len(feature_tokens), desc="Writing features"):
            tok = feature_tokens[idx]
            concept_id = concept_ids[idx] if idx < len(concept_ids) else ''
            cnt = feature_counts[idx]
            f.write(f"{idx}\t{tok}\t{concept_id}\t{cnt}\n")

    print(f"Saved TRAIN X, y, patient_ids, and features to {out_dir}")
    print(f"TRAIN dataset: {len(pids_train)} patients × {len(feature_tokens)} features")
    print(f"TRAIN target distribution: {int(np.sum(y_train))} positive, {len(pids_train) - int(np.sum(y_train))} negative")
    if used_sparse_train:
        print(f"TRAIN feature matrix shape: X={X_train.shape}, y={y_train.shape}")
    else:
        print(f"TRAIN feature matrix shape: X={X_train.shape}, y={y_train.shape}")

    # Optional test set using the same tokenization
    if args.test_data_dir:
        test_out_dir = args.test_out_dir or f"{args.test_data_dir.rstrip(os.sep)}_rf"
        os.makedirs(test_out_dir, exist_ok=True)
        test_timelines = load_patient_timelines_only(args.test_data_dir, allow_events_fallback=True)
        print(f"Building TEST feature matrix for {len(test_timelines)} patients with train tokenization (sparse={use_sparse})")
        X_test, y_test, pids_test, used_sparse_test = build_matrix_for_timelines(
            test_timelines, vocab, feature_id_to_col, target_id_set, args.train_days, args.test_days, use_sparse
        )
        print("Saving TEST outputs...")
        if used_sparse_test:
            sp.save_npz(os.path.join(test_out_dir, 'X_sparse.npz'), X_test)
        else:
            np.save(os.path.join(test_out_dir, 'X.npy'), X_test)
        np.save(os.path.join(test_out_dir, 'y.npy'), y_test)
        np.save(os.path.join(test_out_dir, 'patient_ids.npy'), pids_test)
        print(f"Saved TEST X, y, patient_ids to {test_out_dir}")
        print(f"TEST dataset: {len(pids_test)} patients × {len(feature_tokens)} features")
        print(f"TEST target distribution: {int(np.sum(y_test))} positive, {len(pids_test) - int(np.sum(y_test))} negative")
        print(f"TEST feature matrix shape: X={X_test.shape}, y={y_test.shape}")


if __name__ == '__main__':
    main()


