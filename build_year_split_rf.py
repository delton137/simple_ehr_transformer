#!/usr/bin/env python3
"""
Build sparse X (counts in 2021) and dense y (binary from 2022) using a tokenization spec.

- Uses --tokenization_spec to determine feature tokens and vocabulary mapping
- Scans two OMOP directories: --omop_2021 and --omop_2022
  - Only includes patients present in BOTH years
  - X: count of each feature token from 2021 events
  - y: 1 if any target token (derived from target_concept_id in spec) appears in 2022 events

Outputs to --out_dir:
  - X_sparse.npz (CSR), y.npy (uint8), patient_ids.npy (int64), features.tsv

Progress bars are shown with tqdm.
"""

import os
import argparse
import pickle
import yaml
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict

import numpy as np
from tqdm import tqdm

try:
    import scipy.sparse as sp
except Exception:
    sp = None

try:
    import polars as pl
except Exception:
    pl = None

import pandas as pd


# Table -> (concept_id column name, prefix)
TABLES = {
    "condition_occurrence": ("condition_concept_id", "CONDITION_OCCURRENCE_"),
    "drug_exposure": ("drug_concept_id", "DRUG_EXPOSURE_"),
    "procedure_occurrence": ("procedure_concept_id", "PROCEDURE_OCCURRENCE_"),
    "measurement": ("measurement_concept_id", "MEASUREMENT_"),
    "observation": ("observation_concept_id", "OBSERVATION_"),
}


def load_spec(spec_path: str) -> Tuple[Dict[str, int], List[str], Dict[str, Optional[int]]]:
    """Load tokenization spec and return:
    - vocab_map: token_name -> vocab_id
    - feature_tokens: ordered list of concept-bearing tokens from spec['concepts']
    - token_name_to_concept_id: token_name -> concept_id (for features.tsv)
    """
    with open(spec_path, 'r') as f:
        spec = yaml.safe_load(f)
    vocab_list: List[str] = spec.get('vocabulary', []) or []
    vocab_map: Dict[str, int] = {name: idx for idx, name in enumerate(vocab_list)}
    token_name_to_concept_id: Dict[str, Optional[int]] = {}
    feats: List[str] = []
    for entry in spec.get('concepts', []) or []:
        name = entry.get('token')
        if not name:
            continue
        token_name_to_concept_id[name] = entry.get('concept_id')
        feats.append(name)
    # Deduplicate preserving order
    seen: Set[str] = set()
    feature_tokens = []
    for t in feats:
        if t not in seen:
            seen.add(t)
            feature_tokens.append(t)
    return vocab_map, feature_tokens, token_name_to_concept_id


def target_tokens_from_spec(spec_path: str, target_concept_id: int) -> List[str]:
    with open(spec_path, 'r') as f:
        spec = yaml.safe_load(f)
    out: List[str] = []
    for entry in spec.get('concepts', []) or []:
        try:
            cid = int(entry.get('concept_id'))
        except Exception:
            continue
        if cid == int(target_concept_id):
            tok = entry.get('token')
            if tok:
                out.append(tok)
    # Dedup
    return sorted(set(out))


def list_parquet_files(table_dir: str) -> List[str]:
    if not os.path.isdir(table_dir):
        return []
    return [os.path.join(table_dir, f) for f in os.listdir(table_dir) if f.endswith('.parquet')]


def scan_year_patients(omop_dir: str) -> Set[int]:
    """Return set of patients (person_id) present in any supported table."""
    patients: Set[int] = set()
    for table in TABLES.keys():
        tdir = os.path.join(omop_dir, table)
        files = list_parquet_files(tdir)
        for fp in files:
            try:
                df = pd.read_parquet(fp, columns=['person_id'])
            except Exception:
                continue
            if df is not None and len(df) > 0 and 'person_id' in df.columns:
                patients.update(map(int, df['person_id'].dropna().astype(np.int64).tolist()))
    return patients


def iter_events(omop_dir: str):
    """Yield tuples (person_id, token_name) for all events in supported tables.
    Reads per table per file to keep memory reasonable.
    """
    for table, (cid_col, prefix) in TABLES.items():
        tdir = os.path.join(omop_dir, table)
        files = list_parquet_files(tdir)
        for fp in tqdm(files, desc=f"Scanning {table}", leave=False):
            try:
                df = pd.read_parquet(fp, columns=['person_id', cid_col])
            except Exception:
                continue
            if df is None or len(df) == 0:
                continue
            if 'person_id' not in df.columns or cid_col not in df.columns:
                continue
            # Drop NA and zero concept ids
            df = df[['person_id', cid_col]].dropna()
            if df.empty:
                continue
            # Filter non-positive concept ids if present
            try:
                df = df[(df[cid_col] > 0)]
            except Exception:
                pass
            for pid, cid in zip(df['person_id'].astype(np.int64), df[cid_col].astype(np.int64)):
                yield int(pid), f"{prefix}{int(cid)}"


def main():
    ap = argparse.ArgumentParser(description='Build year-split sparse X (2021) and y (2022) using tokenization spec')
    ap.add_argument('--tokenization_spec', required=True, help='Path to train tokenization.yaml')
    ap.add_argument('--omop_2021', required=True, help='Path to OMOP parquet root for 2021')
    ap.add_argument('--omop_2022', required=True, help='Path to OMOP parquet root for 2022')
    ap.add_argument('--target_concept_id', type=int, default=201826, help='Target concept_id for labels (default: 201826)')
    ap.add_argument('--out_dir', required=True, help='Output directory for matrices')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load spec and derive features + mapping
    vocab_map, feature_tokens, token_to_cid = load_spec(args.tokenization_spec)
    target_tokens = set(target_tokens_from_spec(args.tokenization_spec, args.target_concept_id))

    # Build feature id mapping using spec vocabulary ids to ensure consistency with train tokenization
    # Optionally exclude target tokens from features to avoid leakage
    filtered_feature_tokens = [t for t in feature_tokens if t not in target_tokens]
    feature_ids = []
    for t in filtered_feature_tokens:
        tid = vocab_map.get(t)
        if tid is not None:
            feature_ids.append(tid)
    feature_id_to_col: Dict[int, int] = {tid: i for i, tid in enumerate(feature_ids)}

    print(f"Total feature tokens (excluding targets): {len(feature_id_to_col)}")
    if len(feature_id_to_col) == 0:
        raise SystemExit("No features found from tokenization_spec. Aborting.")

    # Find patients present in both years
    print("Collecting patient sets for 2021 and 2022...")
    patients_2021 = scan_year_patients(args.omop_2021)
    patients_2022 = scan_year_patients(args.omop_2022)
    common_patients = sorted(patients_2021.intersection(patients_2022))
    print(f"Patients: 2021={len(patients_2021)}, 2022={len(patients_2022)}, common={len(common_patients)}")
    if not common_patients:
        raise SystemExit("No overlapping patients between 2021 and 2022.")

    # Map common patient ids to row indices
    pid_to_row: Dict[int, int] = {pid: i for i, pid in enumerate(common_patients)}

    # Build X from 2021: counts of feature tokens
    print("Building 2021 feature counts (X)...")
    rows: List[int] = []
    cols: List[int] = []
    data: List[int] = []
    # Accumulate counts per patient row to compress writes
    per_patient_counts: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for pid, token_name in tqdm(iter_events(args.omop_2021), desc='Events 2021'):
        row = pid_to_row.get(pid)
        if row is None:
            continue
        tid = vocab_map.get(token_name)
        if tid is None:
            continue
        col = feature_id_to_col.get(tid)
        if col is None:
            continue
        per_patient_counts[row][col] += 1
    # Emit to COO lists
    for row, cnts in per_patient_counts.items():
        for col, val in cnts.items():
            rows.append(row); cols.append(col); data.append(int(val))
    if sp is None:
        raise RuntimeError("scipy is required to save sparse matrices. Please install scipy.")
    X_sparse = sp.coo_matrix((data, (rows, cols)), shape=(len(common_patients), len(feature_id_to_col)), dtype=np.int32).tocsr()
    print(f"X shape: {X_sparse.shape}, nnz={X_sparse.nnz}")

    # Build y from 2022: presence of any target token
    print("Building 2022 labels (y) from target presence...")
    target_token_set = set(target_tokens)
    y = np.zeros((len(common_patients),), dtype=np.uint8)
    # Simple presence check per patient
    seen_positive: Set[int] = set()
    for pid, token_name in tqdm(iter_events(args.omop_2022), desc='Events 2022'):
        row = pid_to_row.get(pid)
        if row is None or row in seen_positive:
            continue
        if token_name in target_token_set:
            y[row] = 1
            seen_positive.add(row)

    # Save outputs
    print("Saving outputs...")
    sp.save_npz(os.path.join(args.out_dir, 'X_sparse.npz'), X_sparse)
    np.save(os.path.join(args.out_dir, 'y.npy'), y)
    np.save(os.path.join(args.out_dir, 'patient_ids.npy'), np.array(common_patients, dtype=np.int64))
    with open(os.path.join(args.out_dir, 'features.tsv'), 'w') as f:
        f.write("index\ttoken\tconcept_id\n")
        for idx, t in enumerate(filtered_feature_tokens):
            cid = token_to_cid.get(t)
            f.write(f"{idx}\t{t}\t{cid if cid is not None else ''}\n")
    print(f"Done. Saved to {args.out_dir}")


if __name__ == '__main__':
    main()


