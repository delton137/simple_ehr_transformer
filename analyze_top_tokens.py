#!/usr/bin/env python3
"""
Analyze the most common tokens and map them to human-readable OMOP concepts.

Outputs a table with columns:
- token: token string (e.g., MEASUREMENT_3004249)
- token_id: integer id from vocabulary
- raw_count: number of occurrences in tokenized timelines
- frequency_percent: percentage of total counted tokens
- interpretation: human-readable name (concept_name or derived label)
- concept_id: OMOP concept_id if applicable
- domain_id, vocabulary_id, concept_code: extra OMOP metadata when available

Notes:
- By default, focuses on concept tokens: CONDITION_, DRUG_, PROCEDURE_, MEASUREMENT_, OBSERVATION_, UNIT_
- Can include non-concept tokens with --include_misc
"""

import os
import argparse
import pickle
from typing import Dict, List, Tuple, Optional
from collections import Counter

import pandas as pd
from tqdm import tqdm

from config import data_config


CONCEPT_PREFIXES = (
    "CONDITION_",
    "DRUG_",
    "PROCEDURE_",
    "MEASUREMENT_",
    "OBSERVATION_",
    "UNIT_",
)


def load_processed_data(data_dir: str) -> Tuple[Dict[int, List[int]], Dict[str, int]]:
    with open(os.path.join(data_dir, 'tokenized_timelines.pkl'), 'rb') as f:
        tokenized_timelines = pickle.load(f)
    with open(os.path.join(data_dir, 'vocabulary.pkl'), 'rb') as f:
        vocab = pickle.load(f)
    return tokenized_timelines, vocab


def invert_vocab(vocab: Dict[str, int]) -> Dict[int, str]:
    return {v: k for k, v in vocab.items()}


def count_token_frequencies(tokenized_timelines: Dict[int, List[int]]) -> Counter:
    counts: Counter = Counter()
    print("Counting token frequencies...")
    for tokens in tqdm(tokenized_timelines.values(), desc="Processing timelines", unit="patient"):
        counts.update(tokens)
    return counts


def parse_token(token_str: str) -> Tuple[Optional[str], Optional[int]]:
    """Return (prefix_without_trailing_, concept_id) if token encodes an OMOP concept, else (None, None)."""
    for prefix in CONCEPT_PREFIXES:
        if token_str.startswith(prefix):
            suffix = token_str[len(prefix):]
            try:
                cid = int(suffix)
            except Exception:
                return None, None
            # Normalize prefix label without trailing underscore
            return prefix[:-1], cid
    return None, None


def convert_dbdate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Convert dbdate columns to timestamp to avoid loading errors."""
    for col in df.columns:
        if df[col].dtype == 'object':
            # Check if this might be a dbdate column
            sample_values = df[col].dropna().head(100)
            if len(sample_values) > 0:
                try:
                    # Try to convert to datetime
                    pd.to_datetime(sample_values, errors='coerce')
                    # If successful, convert the entire column
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    print(f"Converted column {col} from dbdate to datetime")
                except:
                    pass
    return df


def read_concept_table(omop_dir: str) -> Optional[pd.DataFrame]:
    concept_dir = os.path.join(omop_dir, 'concept')
    if not os.path.isdir(concept_dir):
        print(f"Concept directory not found: {concept_dir}")
        return None
    files = [os.path.join(concept_dir, f) for f in os.listdir(concept_dir) if f.endswith('.parquet')]
    if not files:
        print(f"No parquet files found in concept directory: {concept_dir}")
        return None
    print(f"Loading concept table from {len(files)} parquet files...")
    dfs: List[pd.DataFrame] = []
    for fp in tqdm(files, desc="Loading concept files", unit="file"):
        try:
            df = pd.read_parquet(fp, engine='pyarrow')
            dfs.append(df)
        except Exception as e:
            print(f"Failed to load {fp} with pyarrow: {e}")
            try:
                df = pd.read_parquet(fp, engine='fastparquet')
                dfs.append(df)
            except Exception as e2:
                print(f"Failed to load {fp} with fastparquet: {e2}")
                # Try with manual dbdate handling
                try:
                    print(f"Attempting to load with manual dbdate conversion...")
                    import pyarrow.parquet as pq
                    table = pq.read_table(fp)
                    # Convert dbdate columns
                    schema = table.schema
                    new_fields = []
                    for field in schema:
                        if str(field.type) == 'dbdate':
                            new_fields.append(pyarrow.field(field.name, pyarrow.timestamp('ns')))
                        else:
                            new_fields.append(field)
                    new_schema = pyarrow.schema(new_fields)
                    # Cast the table
                    table = table.cast(new_schema)
                    df = table.to_pandas()
                    print(f"Successfully loaded with dbdate conversion")
                    dfs.append(df)
                except Exception as e3:
                    print(f"Failed to load with dbdate conversion: {e3}")
                    continue
    if not dfs:
        print("No concept files could be loaded successfully")
        return None
    print("Concatenating concept data...")
    df_all = pd.concat(dfs, ignore_index=True)
    print(f"Concept table shape: {df_all.shape}")
    print(f"Concept table columns: {list(df_all.columns)}")
    
    # Keep relevant columns if present
    keep_cols = [
        'concept_id', 'concept_name', 'domain_id', 'vocabulary_id', 'concept_code',
        'standard_concept'
    ]
    cols = [c for c in keep_cols if c in df_all.columns]
    print(f"Available columns: {cols}")
    if not cols:
        print("Warning: No expected columns found in concept table")
        return None
    
    result = df_all[cols].drop_duplicates('concept_id') if 'concept_id' in cols else df_all.drop_duplicates()
    print(f"Final concept table shape: {result.shape}")
    return result


def read_concept_relationship_table(omop_dir: str) -> Optional[pd.DataFrame]:
    rel_dir = os.path.join(omop_dir, 'concept_relationship')
    if not os.path.isdir(rel_dir):
        print(f"Concept relationship directory not found: {rel_dir}")
        return None
    files = [os.path.join(rel_dir, f) for f in os.listdir(rel_dir) if f.endswith('.parquet')]
    if not files:
        print(f"No parquet files found in concept relationship directory: {rel_dir}")
        return None
    print(f"Loading concept relationship table from {len(files)} parquet files...")
    dfs: List[pd.DataFrame] = []
    for fp in tqdm(files, desc="Loading relationship files", unit="file"):
        try:
            df = pd.read_parquet(fp, engine='pyarrow')
            dfs.append(df)
        except Exception as e:
            print(f"Failed to load {fp} with pyarrow: {e}")
            try:
                df = pd.read_parquet(fp, engine='fastparquet')
                dfs.append(df)
            except Exception as e2:
                print(f"Failed to load {fp} with fastparquet: {e2}")
                # Try with manual dbdate handling
                try:
                    print(f"Attempting to load with manual dbdate conversion...")
                    import pyarrow.parquet as pq
                    table = pq.read_table(fp)
                    # Convert dbdate columns
                    schema = table.schema
                    new_fields = []
                    for field in schema:
                        if str(field.type) == 'dbdate':
                            new_fields.append(pyarrow.field(field.name, pyarrow.timestamp('ns')))
                        else:
                            new_fields.append(field)
                    new_schema = pyarrow.schema(new_fields)
                    # Cast the table
                    table = table.cast(new_schema)
                    df = table.to_pandas()
                    print(f"Successfully loaded with dbdate conversion")
                    dfs.append(df)
                except Exception as e3:
                    print(f"Failed to load with dbdate conversion: {e3}")
                    continue
    if not dfs:
        print("No relationship files could be loaded successfully")
        return None
    print("Concatenating relationship data...")
    df_all = pd.concat(dfs, ignore_index=True)
    print(f"Relationship table shape: {df_all.shape}")
    print(f"Relationship table columns: {list(df_all.columns)}")
    
    # Normalize column names
    expected = {'concept_id_1', 'concept_id_2', 'relationship_id'}
    if not expected.issubset(set(df_all.columns)):
        print(f"Warning: Relationship table missing expected columns. Expected: {expected}, Found: {set(df_all.columns)}")
        return None
    return df_all[['concept_id_1', 'concept_id_2', 'relationship_id']]


def map_to_standard_concept(concept_ids: List[int], concept_df: Optional[pd.DataFrame], rel_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    print(f"Mapping {len(concept_ids)} concepts to standard concepts...")
    ci_df = pd.DataFrame({'concept_id': concept_ids}).drop_duplicates()
    if rel_df is not None:
        print("Applying 'Maps to' relationships...")
        maps_to = rel_df[rel_df['relationship_id'] == 'Maps to'][['concept_id_1', 'concept_id_2']].drop_duplicates()
        ci_df = ci_df.merge(maps_to, how='left', left_on='concept_id', right_on='concept_id_1')
        ci_df['standard_id'] = ci_df['concept_id_2'].fillna(ci_df['concept_id'])
        ci_df = ci_df.drop(columns=['concept_id_1', 'concept_id_2'])
    else:
        ci_df['standard_id'] = ci_df['concept_id']
    if concept_df is not None:
        print("Joining concept metadata...")
        concept_cols = ['concept_id', 'concept_name', 'domain_id', 'vocabulary_id', 'concept_code']
        # Join on standard_id for final interpretation
        ci_df = ci_df.merge(concept_df.rename(columns={'concept_id': 'standard_id'}), how='left', on='standard_id')
        # For completeness, also provide original concept metadata if different
        base_meta_cols = [c for c in concept_cols if c in concept_df.columns]
        if base_meta_cols:
            ci_df = ci_df.merge(concept_df[base_meta_cols].rename(columns={c: f'orig_{c}' for c in base_meta_cols}),
                                how='left', left_on='concept_id', right_on=f'orig_concept_id')
    return ci_df


def build_top_table(
    counts: Counter,
    id_to_token: Dict[int, str],
    concept_df: Optional[pd.DataFrame],
    rel_df: Optional[pd.DataFrame],
    top_k: int,
    concept_only: bool,
) -> pd.DataFrame:
    print(f"Building top {top_k} tokens table...")
    # Build list of (token_id, token_str, count)
    rows: List[Tuple[int, str, int]] = []
    for tid, cnt in tqdm(counts.items(), desc="Processing token counts", unit="token"):
        token_str = id_to_token.get(tid)
        if token_str is None:
            continue
        if concept_only and not token_str.startswith(CONCEPT_PREFIXES):
            continue
        rows.append((tid, token_str, cnt))
    if not rows:
        return pd.DataFrame(columns=['token', 'token_id', 'raw_count', 'frequency_percent', 'interpretation', 'concept_id', 'domain_id', 'vocabulary_id', 'concept_code'])

    print("Creating DataFrame and calculating frequencies...")
    df = pd.DataFrame(rows, columns=['token_id', 'token', 'raw_count'])
    df = df.sort_values('raw_count', ascending=False)
    total = df['raw_count'].sum()
    df['frequency_percent'] = (df['raw_count'] / max(1, total)) * 100.0

    # Extract OMOP concept ids when applicable
    print("Extracting concept IDs from tokens...")
    parsed = df['token'].apply(parse_token)
    df['prefix'] = parsed.apply(lambda x: x[0])
    df['concept_id'] = parsed.apply(lambda x: x[1])

    # Map concept ids to standard concepts and names
    has_concepts = df['concept_id'].notna()
    if has_concepts.any() and concept_df is not None:
        concept_ids = df.loc[has_concepts, 'concept_id'].astype(int).tolist()
        mapped = map_to_standard_concept(concept_ids, concept_df, rel_df)
        # Check if mapped has the expected columns before merging
        expected_cols = ['concept_id', 'standard_id', 'concept_name', 'domain_id', 'vocabulary_id', 'concept_code']
        available_cols = [col for col in expected_cols if col in mapped.columns]
        if len(available_cols) >= 2:  # Need at least concept_id and one other column
            # Merge back on concept_id
            df = df.merge(mapped[available_cols], how='left', on='concept_id')
            # Interpretation preference: concept_name if present
            if 'concept_name' in available_cols:
                df['interpretation'] = df['concept_name']
            else:
                df['interpretation'] = None
        else:
            print(f"Warning: Mapped concept table missing expected columns. Available: {list(mapped.columns)}")
            df['interpretation'] = None
    else:
        df['interpretation'] = None

    # For tokens without OMOP concept, provide a readable interpretation
    missing_interp = df['interpretation'].isna()
    if missing_interp.any():
        print("Adding fallback interpretations for non-concept tokens...")
        def fallback_interp(tok: str) -> str:
            if tok.startswith('AGE_'):
                return f"Age interval {tok[len('AGE_') : ]} years"
            if tok.startswith('TIME_'):
                return f"Time gap {tok[len('TIME_') : ]}"
            if tok.startswith('EVENT_'):
                return tok[len('EVENT_') : ].capitalize()
            if tok.startswith('GENDER_'):
                return f"Gender concept {tok[len('GENDER_') : ]}"
            if tok.startswith('RACE_'):
                return f"Race concept {tok[len('RACE_') : ]}"
            if tok.startswith('YEAR_'):
                return f"Birth year interval {tok[len('YEAR_') : ]}"
            if tok.startswith('Q') and tok[1:].isdigit():
                return f"Quantile token {tok}"
            if tok in ('<PAD>', '<UNK>', '<EOS>', '<SOS>'):
                return tok
            return tok

        df.loc[missing_interp, 'interpretation'] = df.loc[missing_interp, 'token'].apply(fallback_interp)

    # Final sort and trim to top_k
    print(f"Finalizing top {top_k} tokens...")
    # Final column selection - only include columns that exist
    available_cols = ['token', 'token_id', 'raw_count', 'frequency_percent', 'interpretation']
    if 'concept_id' in df.columns:
        available_cols.append('concept_id')
    if 'domain_id' in df.columns:
        available_cols.append('domain_id')
    if 'vocabulary_id' in df.columns:
        available_cols.append('vocabulary_id')
    if 'concept_code' in df.columns:
        available_cols.append('concept_code')
    
    df = df[available_cols]
    df = df.sort_values('raw_count', ascending=False).head(top_k).reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description='Find the most common tokens and map to OMOP concepts')
    parser.add_argument('--data_dir', type=str, default=None, help='Processed data directory (default: processed_data or processed_data_{tag})')
    parser.add_argument('--tag', type=str, default=None, help='Dataset tag to locate processed_data_{tag}')
    parser.add_argument('--omop_dir', type=str, default=None, help='OMOP data directory containing concept/ and concept_relationship/')
    parser.add_argument('--top_k', type=int, default=100, help='Number of top tokens to report (default: 100)')
    parser.add_argument('--include_misc', action='store_true', help='Include non-concept tokens (EVENT_/AGE_/TIME_/etc.)')
    parser.add_argument('--output_csv', type=str, default=None, help='Path to save CSV (default: {data_dir}/top_tokens.csv)')

    args = parser.parse_args()

    # Resolve data_dir
    data_dir = args.data_dir
    if data_dir is None:
        data_dir = f"processed_data_{args.tag}" if args.tag else 'processed_data'

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Processed data directory not found: {data_dir}")

    # Resolve omop_dir
    omop_dir = args.omop_dir or data_config.omop_data_dir

    # Load processed data
    print("Loading processed data...")
    tokenized_timelines, vocab = load_processed_data(data_dir)
    id_to_token = invert_vocab(vocab)
    print(f"Loaded {len(tokenized_timelines)} patient timelines with {len(vocab)} vocabulary tokens")
    
    print("Counting token frequencies...")
    counts = count_token_frequencies(tokenized_timelines)

    # Load OMOP concept tables (best-effort)
    print("Loading OMOP concept tables...")
    concept_df = read_concept_table(omop_dir)
    rel_df = read_concept_relationship_table(omop_dir)
    
    if concept_df is not None:
        print(f"Loaded concept table with {len(concept_df)} concepts")
    if rel_df is not None:
        print(f"Loaded relationship table with {len(rel_df)} relationships")
    if concept_df is None and rel_df is None:
        print("No OMOP concept tables found - will use fallback interpretations")

    # Build table
    table = build_top_table(
        counts=counts,
        id_to_token=id_to_token,
        concept_df=concept_df,
        rel_df=rel_df,
        top_k=args.top_k,
        concept_only=(not args.include_misc),
    )

    # Output CSV
    out_csv = args.output_csv or os.path.join(data_dir, 'top_tokens.csv')
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    table.to_csv(out_csv, index=False)

    # Print a brief preview
    print(f"Saved top {len(table)} tokens to {out_csv}")
    try:
        preview = table.head(min(20, len(table)))
        with pd.option_context('display.max_colwidth', 80):
            print(preview)
    except Exception:
        pass


if __name__ == '__main__':
    main()


