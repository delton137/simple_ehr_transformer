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

import pandas as pd
import numpy as np
from collections import Counter
from typing import Dict, List, Optional, Tuple
import argparse
import os
import json
import pickle
from tqdm import tqdm
import polars as pl
import pyarrow
from pathlib import Path

from config import data_config


def get_concept_name(
    table: str,
    concept_id: int,
    concept_parquet_path: str | Path
) -> str | None:
    """
    Look up the OMOP concept_name for a given concept_id using Polars.

    Parameters
    ----------
    table : str
        Name of the OMOP table (not used for lookup, but included for clarity/logging).
    concept_id : int
        The concept_id to look up.
    concept_parquet_path : str | Path
        Path to the OMOP concept table parquet file.

    Returns
    -------
    str | None
        The concept_name if found, else None.
    """
    try:
        # Load only needed columns to save memory
        concept_df = pl.read_parquet(concept_parquet_path, columns=["concept_id", "concept_name"])

        result = (
            concept_df
            .filter(pl.col("concept_id") == concept_id)
            .select("concept_name")
            .to_series()
        )

        if result.is_empty():
            print(f"[WARN] concept_id {concept_id} not found in {table}.")
            return None
        
        # Return as plain Python string
        return result[0]
    except Exception as e:
        print(f"[ERROR] Failed to lookup concept_id {concept_id} in {table}: {e}")
        return None


def get_concept_names_batch(
    concept_codes: List[int],
    concept_parquet_path: str | Path
) -> Dict[int, str]:
    """
    Batch lookup concept names for multiple concept codes.
    
    Parameters
    ----------
    concept_codes : List[int]
        List of concept codes to look up.
    concept_parquet_path : str | Path
        Path to the OMOP concept table parquet file.
        
    Returns
    -------
    Dict[int, str]
        Mapping from concept_code to concept_name.
    """
    try:
        # Load only needed columns to save memory
        concept_df = pl.read_parquet(concept_parquet_path, columns=["concept_id", "concept_name"])
        
        # Filter to only the concept codes we need
        result_df = (
            concept_df
            .filter(pl.col("concept_id").is_in(concept_codes))
            .select(["concept_id", "concept_name"])
        )
        
        # Convert to dictionary
        concept_names = {}
        for row in result_df.iter_rows():
            concept_names[row[0]] = row[1]
        
        print(f"Successfully mapped {len(concept_names)} out of {len(concept_codes)} concept codes")
        return concept_names
        
    except Exception as e:
        print(f"[ERROR] Failed to batch lookup concepts: {e}")
        return {}


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
    print(f"Looking for concept directory: {concept_dir}")
    if not os.path.isdir(concept_dir):
        print(f"❌ Concept directory not found: {concept_dir}")
        print(f"Available directories in {omop_dir}: {os.listdir(omop_dir) if os.path.exists(omop_dir) else 'Directory does not exist'}")
        return None
    
    files = [os.path.join(concept_dir, f) for f in os.listdir(concept_dir) if f.endswith('.parquet')]
    if not files:
        print(f"❌ No parquet files found in concept directory: {concept_dir}")
        print(f"Files in concept directory: {os.listdir(concept_dir)}")
        return None
    
    print(f"Loading concept table from {len(files)} parquet files...")
    print(f"Files: {files}")
    dfs: List[pd.DataFrame] = []
    
    for fp in tqdm(files, desc="Loading concept files", unit="file"):
        print(f"Attempting to load: {fp}")
        try:
            df = pd.read_parquet(fp, engine='pyarrow')
            print(f"✅ Successfully loaded with pyarrow: {df.shape}")
            dfs.append(df)
        except Exception as e:
            print(f"❌ Failed to load {fp} with pyarrow: {e}")
            try:
                df = pd.read_parquet(fp, engine='fastparquet')
                print(f"✅ Successfully loaded with fastparquet: {df.shape}")
                dfs.append(df)
            except Exception as e2:
                print(f"❌ Failed to load {fp} with fastparquet: {e2}")
                # Try with manual dbdate handling
                try:
                    print(f"Attempting to load with manual dbdate conversion...")
                    import pyarrow.parquet as pq
                    table = pq.read_table(fp)
                    print(f"✅ Successfully read with pyarrow.parquet: {table.shape}")
                    print(f"Schema: {table.schema}")
                    
                    # Convert dbdate columns
                    schema = table.schema
                    new_fields = []
                    for field in schema:
                        if str(field.type) == 'dbdate':
                            new_fields.append(pyarrow.field(field.name, pyarrow.timestamp('ns')))
                            print(f"Converting dbdate field: {field.name}")
                        else:
                            new_fields.append(field)
                    
                    if new_fields != list(schema):
                        new_schema = pyarrow.schema(new_fields)
                        print(f"New schema: {new_schema}")
                        # Cast the table
                        table = table.cast(new_schema)
                        print(f"✅ Successfully cast table")
                    
                    df = table.to_pandas()
                    print(f"✅ Successfully converted to pandas: {df.shape}")
                    dfs.append(df)
                except Exception as e3:
                    print(f"❌ Failed to load with dbdate conversion: {e3}")
                    import traceback
                    print("Full stack trace for dbdate conversion:")
                    traceback.print_exc()
                    continue
    
    if not dfs:
        print("❌ No concept files could be loaded successfully")
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
        print("❌ Warning: No expected columns found in concept table")
        print(f"All columns: {list(df_all.columns)}")
        return None
    
    result = df_all[cols].drop_duplicates('concept_id') if 'concept_id' in cols else df_all.drop_duplicates()
    print(f"Final concept table shape: {result.shape}")
    return result


def read_concept_relationship_table(omop_dir: str) -> Optional[pd.DataFrame]:
    rel_dir = os.path.join(omop_dir, 'concept_relationship')
    print(f"Looking for relationship directory: {rel_dir}")
    if not os.path.isdir(rel_dir):
        print(f"❌ Concept relationship directory not found: {rel_dir}")
        print(f"Available directories in {omop_dir}: {os.listdir(omop_dir) if os.path.exists(omop_dir) else 'Directory does not exist'}")
        return None
    
    files = [os.path.join(rel_dir, f) for f in os.listdir(rel_dir) if f.endswith('.parquet')]
    if not files:
        print(f"❌ No parquet files found in concept relationship directory: {rel_dir}")
        print(f"Files in relationship directory: {os.listdir(rel_dir)}")
        return None
    
    print(f"Loading concept relationship table from {len(files)} parquet files...")
    print(f"Files: {files}")
    dfs: List[pd.DataFrame] = []
    
    for fp in tqdm(files, desc="Loading relationship files", unit="file"):
        print(f"Attempting to load: {fp}")
        try:
            df = pd.read_parquet(fp, engine='pyarrow')
            print(f"✅ Successfully loaded with pyarrow: {df.shape}")
            dfs.append(df)
        except Exception as e:
            print(f"❌ Failed to load {fp} with pyarrow: {e}")
            try:
                df = pd.read_parquet(fp, engine='fastparquet')
                print(f"✅ Successfully loaded with fastparquet: {df.shape}")
                dfs.append(df)
            except Exception as e2:
                print(f"❌ Failed to load {fp} with fastparquet: {e2}")
                # Try with manual dbdate handling
                try:
                    print(f"Attempting to load with manual dbdate conversion...")
                    import pyarrow.parquet as pq
                    table = pq.read_table(fp)
                    print(f"✅ Successfully read with pyarrow.parquet: {table.shape}")
                    print(f"Schema: {table.schema}")
                    
                    # Convert dbdate columns
                    schema = table.schema
                    new_fields = []
                    for field in schema:
                        if str(field.type) == 'dbdate':
                            new_fields.append(pyarrow.field(field.name, pyarrow.timestamp('ns')))
                            print(f"Converting dbdate field: {field.name}")
                        else:
                            new_fields.append(field)
                    
                    if new_fields != list(schema):
                        new_schema = pyarrow.schema(new_fields)
                        print(f"New schema: {new_schema}")
                        # Cast the table
                        table = table.cast(new_schema)
                        print(f"✅ Successfully cast table")
                    
                    # Handle the dbdate issue in schema metadata by creating a clean schema
                    try:
                        df = table.to_pandas()
                        print(f"✅ Successfully converted to pandas: {df.shape}")
                        dfs.append(df)
                    except Exception as pandas_error:
                        if "dbdate" in str(pandas_error).lower():
                            print("⚠️  dbdate issue in pandas conversion, trying alternative approach...")
                            # Create a new table with clean schema metadata
                            clean_schema = pyarrow.schema([
                                pyarrow.field(field.name, field.type) 
                                for field in table.schema
                            ])
                            clean_table = table.cast(clean_schema)
                            df = clean_table.to_pandas()
                            print(f"✅ Successfully converted to pandas with clean schema: {df.shape}")
                            dfs.append(df)
                        else:
                            raise pandas_error
                            
                except Exception as e3:
                    print(f"❌ Failed to load with dbdate conversion: {e3}")
                    import traceback
                    print("Full stack trace for dbdate conversion:")
                    traceback.print_exc()
                    continue
    
    if not dfs:
        print("❌ No relationship files could be loaded successfully")
        return None
    
    print("Concatenating relationship data...")
    df_all = pd.concat(dfs, ignore_index=True)
    print(f"Relationship table shape: {df_all.shape}")
    print(f"Relationship table columns: {list(df_all.columns)}")
    
    # Normalize column names
    expected = {'concept_id_1', 'concept_id_2', 'relationship_id'}
    if not expected.issubset(set(df_all.columns)):
        print(f"❌ Warning: Relationship table missing expected columns. Expected: {expected}, Found: {set(df_all.columns)}")
        return None
    
    result = df_all[['concept_id_1', 'concept_id_2', 'relationship_id']]
    print(f"Final relationship table shape: {result.shape}")
    return result


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
        # Use the actual columns from your OMOP concept table
        concept_cols = ['concept_id', 'concept_name', 'domain_id', 'vocabulary_id', 'concept_code', 'concept_class_id', 'standard_concept']
        available_concept_cols = [c for c in concept_cols if c in concept_df.columns]
        print(f"Available concept columns: {available_concept_cols}")
        
        if available_concept_cols:
            # Join original concept metadata on concept_id
            ci_df = ci_df.merge(concept_df[available_concept_cols], how='left', on='concept_id')
            
            # If we have standard concepts, also get their metadata
            if 'standard_id' in ci_df.columns and 'standard_id' != 'concept_id':
                standard_meta = concept_df[available_concept_cols].rename(columns={c: f'standard_{c}' for c in available_concept_cols})
                ci_df = ci_df.merge(standard_meta, how='left', left_on='standard_id', right_on='standard_concept_id')
                
                # Prefer standard concept names if available
                if 'standard_concept_name' in ci_df.columns:
                    ci_df['concept_name'] = ci_df['standard_concept_name'].fillna(ci_df['concept_name'])
    else:
        print("No concept table available for metadata")
    
    return ci_df


def build_top_table(
    counts: Counter,
    id_to_token: Dict[int, str],
    concept_parquet_path: Optional[str],
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

    # Extract OMOP concept codes when applicable
    print("Extracting concept codes from tokens...")
    parsed = df['token'].apply(parse_token)
    df['prefix'] = parsed.apply(lambda x: x[0])
    df['concept_code'] = parsed.apply(lambda x: x[1])  # This is actually the concept_code, not concept_id
    
    # Initialize concept_name and interpretation columns
    df['concept_name'] = None
    df['interpretation'] = None
    
    # Map concept codes to concept names and metadata using OMOP concept table
    has_concepts = df['concept_code'].notna()
    if has_concepts.any() and concept_parquet_path:
        print(f"Using Polars-based concept lookup for top {len(df)} tokens...")
        concept_codes = df.loc[has_concepts, 'concept_code'].astype(int).tolist()
        print(f"Found {len(concept_codes)} concept codes to map (only for top {len(df)} tokens)")
        
        # Use batch lookup for efficiency
        print("Looking up concept names in batch...")
        concept_names_dict = get_concept_names_batch(concept_codes, concept_parquet_path)
        
        # Map concept names back to DataFrame
        df['concept_name'] = df['concept_code'].map(concept_names_dict)
        
        # Set interpretation to concept_name when available
        df['interpretation'] = df['concept_name'].fillna(df['interpretation'])
        
        # Try to enrich with additional concept metadata if available
        try:
            print("Enriching with additional concept metadata...")
            import polars as pl
            
            # Read concept table to get additional metadata
            concept_df = pl.read_parquet(concept_parquet_path, columns=["concept_id", "concept_name", "domain_id", "vocabulary_id", "concept_class_id"])
            
            # Convert to pandas for easier merging
            concept_pandas = concept_df.to_pandas()
            
            # Merge with our DataFrame to get additional metadata
            # Note: concept_code in our DataFrame corresponds to concept_id in OMOP
            df_enriched = df.merge(
                concept_pandas, 
                how='left', 
                left_on='concept_code', 
                right_on='concept_id'
            )
            
            # Add the new columns if they don't exist
            if 'domain_id' in df_enriched.columns:
                df['domain_id'] = df_enriched['domain_id']
            if 'vocabulary_id' in df_enriched.columns:
                df['vocabulary_id'] = df_enriched['vocabulary_id']
            if 'concept_class_id' in df_enriched.columns:
                df['concept_class_id'] = df_enriched['concept_class_id']
            
            print(f"Successfully enriched {df_enriched['domain_id'].notna().sum()} concepts with metadata")
            
        except Exception as e:
            print(f"⚠️  Warning: Failed to enrich with concept metadata: {e}")
            print("   Continuing with basic concept names only")
        
        print(f"Successfully mapped concept names for {df['concept_name'].notna().sum()} concepts")

    # For tokens without OMOP concept, provide a readable interpretation
    missing_interp = df['interpretation'].isna()
    if missing_interp.any():
        print("Adding fallback interpretations for non-concept tokens...")
        def fallback_interp(tok: str) -> str:
            # Special tokens
            if tok in ('<PAD>', '<UNK>', '<EOS>', '<SOS>'):
                return tok
            
            # Event types
            if tok.startswith('EVENT_'):
                event_type = tok[len('EVENT_'):].lower()
                event_map = {
                    'condition': 'Medical Condition',
                    'drug': 'Medication',
                    'procedure': 'Medical Procedure', 
                    'measurement': 'Lab Test/Measurement',
                    'observation': 'Clinical Observation',
                    'visit': 'Healthcare Visit',
                    'death': 'Death Event'
                }
                return event_map.get(event_type, f"Medical Event: {event_type.title()}")
            
            # Age intervals
            if tok.startswith('AGE_'):
                age_range = tok[len('AGE_'):]
                if age_range == '0':
                    return "Age: 0-1 years"
                elif age_range == '1':
                    return "Age: 1-5 years"
                elif age_range == '5':
                    return "Age: 5-10 years"
                elif age_range == '10':
                    return "Age: 10-15 years"
                elif age_range == '15':
                    return "Age: 15-20 years"
                elif age_range == '20':
                    return "Age: 20-30 years"
                elif age_range == '30':
                    return "Age: 30-40 years"
                elif age_range == '40':
                    return "Age: 40-50 years"
                elif age_range == '50':
                    return "Age: 50-60 years"
                elif age_range == '60':
                    return "Age: 60-70 years"
                elif age_range == '70':
                    return "Age: 70-80 years"
                elif age_range == '80':
                    return "Age: 80+ years"
                else:
                    return f"Age interval: {age_range} years"
            
            # Time intervals
            if tok.startswith('TIME_'):
                time_gap = tok[len('TIME_'):]
                if time_gap == '0':
                    return "Time: Same day"
                elif time_gap == '1':
                    return "Time: 1-6 months gap"
                elif time_gap == '2':
                    return "Time: 6-12 months gap"
                elif time_gap == '3':
                    return "Time: 1-2 years gap"
                elif time_gap == '4':
                    return "Time: 2-5 years gap"
                elif time_gap == '5':
                    return "Time: 5+ years gap"
                else:
                    return f"Time gap: {time_gap} intervals"
            
            # Gender
            if tok.startswith('GENDER_'):
                gender = tok[len('GENDER_'):]
                gender_map = {'0': 'Unknown', '1': 'Male', '2': 'Female'}
                return f"Gender: {gender_map.get(gender, gender)}"
            
            # Race/Ethnicity
            if tok.startswith('RACE_'):
                race = tok[len('RACE_'):]
                race_map = {
                    '0': 'Unknown', '1': 'White', '2': 'Black/African American',
                    '3': 'Asian', '4': 'Hispanic/Latino', '5': 'Other'
                }
                return f"Race: {race_map.get(race, race)}"
            
            # Birth year intervals
            if tok.startswith('YEAR_'):
                year_range = tok[len('YEAR_'):]
                if year_range == '0':
                    return "Birth year: 2000+"
                elif year_range == '1':
                    return "Birth year: 1990-1999"
                elif year_range == '2':
                    return "Birth year: 1980-1989"
                elif year_range == '3':
                    return "Birth year: 1970-1979"
                elif year_range == '4':
                    return "Birth year: 1960-1969"
                elif year_range == '5':
                    return "Birth year: 1950-1959"
                elif year_range == '6':
                    return "Birth year: 1940-1949"
                elif year_range == '7':
                    return "Birth year: 1930-1939"
                elif year_range == '8':
                    return "Birth year: 1920-1929"
                elif year_range == '9':
                    return "Birth year: 1910-1919"
                elif year_range == '10':
                    return "Birth year: 1900-1909"
                else:
                    return f"Birth year interval: {year_range}"
            
            # Visit types
            if tok.startswith('VISIT_TYPE_'):
                visit_type = tok[len('VISIT_TYPE_'):]
                visit_map = {
                    '0': 'Unknown', '1': 'Inpatient', '2': 'Outpatient',
                    '3': 'Emergency', '4': 'Urgent Care', '5': 'Primary Care'
                }
                return f"Visit type: {visit_map.get(visit_type, visit_type)}"
            
            # Units
            if tok.startswith('UNIT_'):
                unit_id = tok[len('UNIT_'):]
                unit_map = {
                    '0': 'Unknown unit', '1': 'mg/dL', '2': 'mmol/L', '3': 'mg/L',
                    '4': 'ng/mL', '5': 'pg/mL', '6': 'U/L', '7': 'mEq/L',
                    '8': 'mmHg', '9': 'bpm', '10': 'kg', '11': 'cm',
                    '12': 'Fahrenheit', '13': 'Celsius', '14': 'inches',
                    '15': 'pounds', '16': 'percent', '17': 'ratio'
                }
                return f"Unit: {unit_map.get(unit_id, f'Unit ID {unit_id}')}"
            
            # Quantiles
            if tok.startswith('Q') and tok[1:].isdigit():
                q_num = int(tok[1:])
                if q_num == 0:
                    return "Quantile: Minimum (0th percentile)"
                elif q_num == 25:
                    return "Quantile: 25th percentile"
                elif q_num == 50:
                    return "Quantile: Median (50th percentile)"
                elif q_num == 75:
                    return "Quantile: 75th percentile"
                elif q_num == 100:
                    return "Quantile: Maximum (100th percentile)"
                else:
                    return f"Quantile: {q_num}th percentile"
            
            # Default fallback
            return f"Token: {tok}"

        df.loc[missing_interp, 'interpretation'] = df.loc[missing_interp, 'token'].apply(fallback_interp)

    # Final sort and trim to top_k
    print(f"Finalizing top {top_k} tokens...")
    # Final column selection - only include columns that exist
    available_cols = ['token', 'token_id', 'raw_count', 'frequency_percent', 'interpretation']
    if 'concept_code' in df.columns:
        available_cols.append('concept_code')
    if 'concept_name' in df.columns:
        available_cols.append('concept_name')
    if 'domain_id' in df.columns:
        available_cols.append('domain_id')
    if 'vocabulary_id' in df.columns:
        available_cols.append('vocabulary_id')
    if 'concept_class_id' in df.columns:
        available_cols.append('concept_class_id')
    
    df = df[available_cols]
    df = df.sort_values('raw_count', ascending=False).head(top_k).reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser(description='Find the most common tokens and map to OMOP concepts')
    parser.add_argument('--data_dir', type=str, default=None, help='Processed data directory (default: processed_data or processed_data_{tag})')
    parser.add_argument('--tag', type=str, default=None, help='Dataset tag to locate processed_data_{tag}')
    parser.add_argument('--omop_dir', type=str, default=None, help='OMOP data directory containing concept/ and concept_relationship/')
    parser.add_argument('--top_k', type=int, default=1000, help='Number of top tokens to report (default: 1000)')
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
    
    # For All of Us data, we need the concept table from omop_data_2023
    # but the processed data might be from a different directory
    if omop_dir and 'omop_data_one_month' in omop_dir:
        # Try to find the concept table in the parent directory
        parent_dir = os.path.dirname(omop_dir)
        concept_candidate = os.path.join(parent_dir, 'omop_data_2023', 'concept')
        if os.path.isdir(concept_candidate):
            print(f"ℹ️  Found concept table in parent directory: {concept_candidate}")
            omop_dir = os.path.join(parent_dir, 'omop_data_2023')
        else:
            print(f"⚠️  No concept table found in {concept_candidate}")
            print(f"   You may need to specify --omop_dir to point to a directory containing concept/ and concept_relationship/ tables")
    
    if not omop_dir:
        print("⚠️  No OMOP directory specified - will use fallback interpretations only")
        print("   Use --omop_dir to specify path to OMOP data with concept tables")

    # Load processed data
    print("Loading processed data...")
    tokenized_timelines, vocab = load_processed_data(data_dir)
    id_to_token = invert_vocab(vocab)
    print(f"Loaded {len(tokenized_timelines)} patient timelines with {len(vocab)} vocabulary tokens")
    
    print("Counting token frequencies...")
    counts = count_token_frequencies(tokenized_timelines)

    # Try to find concept table parquet file
    concept_parquet_path = None
    if omop_dir:
        concept_dir = os.path.join(omop_dir, 'concept')
        if os.path.isdir(concept_dir):
            concept_files = [f for f in os.listdir(concept_dir) if f.endswith('.parquet')]
            if concept_files:
                concept_parquet_path = os.path.join(concept_dir, concept_files[0])
                print(f"✅ Found concept table: {concept_parquet_path}")
            else:
                print("⚠️  No concept parquet files found in concept directory")
        else:
            print("⚠️  Concept directory not found")
    
    # Fallback: try the hardcoded path from user's example
    if concept_parquet_path is None:
        fallback_path = "/home/jupyter/workspaces/ehrtransformerbaseline/omop_data_2023/concept/000000000000.parquet"
        if os.path.exists(fallback_path):
            concept_parquet_path = fallback_path
            print(f"✅ Using fallback concept table: {concept_parquet_path}")
        else:
            print(f"⚠️  Fallback concept table not found: {fallback_path}")
    
    if concept_parquet_path:
        print("ℹ️  Will use Polars-based concept lookup for efficient mapping")
    else:
        print("ℹ️  No concept table available - will use fallback interpretations only")
    
    # Try to load relationship table if available (optional - not needed for basic concept mapping)
    rel_df = None
    if omop_dir and os.path.exists(os.path.join(omop_dir, 'concept_relationship')):
        try:
            rel_df = read_concept_relationship_table(omop_dir)
            if rel_df is not None:
                print(f"✅ Successfully loaded relationship table with {len(rel_df)} relationships")
                print(f"   Available columns: {list(rel_df.columns)}")
            else:
                print("⚠️  No relationship table available")
        except Exception as e:
            print(f"⚠️  Failed to load relationship table: {e}")
            print("Will continue without relationship data")
    else:
        print("ℹ️  No relationship table found - will continue without it")
    
    if concept_parquet_path is None and rel_df is None:
        print("ℹ️  Running with fallback interpretations only - no OMOP concept data available")

    # Build table
    table = build_top_table(
        counts=counts,
        id_to_token=id_to_token,
        concept_parquet_path=concept_parquet_path,
        rel_df=rel_df,
        top_k=args.top_k,
        concept_only=not args.include_misc,
    )

    # Display preview
    print(f"\n📋 Top {min(20, len(table))} Tokens Preview:")
    preview_cols = ['token', 'raw_count', 'frequency_percent', 'interpretation']
    available_preview_cols = [col for col in preview_cols if col in table.columns]
    
    with pd.option_context('display.max_colwidth', 50, 'display.width', 120):
        print(table[available_preview_cols].head(20).to_string(index=False))
    
    if len(table) > 20:
        print(f"... and {len(table) - 20} more tokens")

    # Save to CSV
    output_csv = args.output_csv
    if output_csv is None:
        output_csv = os.path.join(data_dir, 'top_tokens.csv')
    
    print(f"\n💾 Saving top {args.top_k} tokens to: {output_csv}")
    table.to_csv(output_csv, index=False)
    print(f"✅ Successfully saved to: {output_csv}")
    
    # Display summary
    print(f"\n📊 Top {args.top_k} Tokens Summary:")
    print(f"   Total unique tokens: {len(counts)}")
    print(f"   Total token occurrences: {sum(counts.values()):,}")
    print(f"   Top token: {table.iloc[0]['token']} ({table.iloc[0]['raw_count']:,} occurrences, {table.iloc[0]['frequency_percent']:.1f}%)")
    
    if 'concept_name' in table.columns and table['concept_name'].notna().any():
        concept_count = table['concept_name'].notna().sum()
        print(f"   Concepts with names: {concept_count}/{args.top_k}")
    
    print(f"\n🎯 Table saved to: {output_csv}")
    print(f"📋 You can now use this CSV for your slides/presentations!")


if __name__ == '__main__':
    main()


