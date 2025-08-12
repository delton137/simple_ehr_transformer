#!/usr/bin/env python3
"""
Data processor for converting OMOP data to tokenized Patient Health Timelines (PHTs)
Optimized for large datasets (several GB) with memory management and chunked processing
Based on the ETHOS paper methodology
"""

import os
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import pyarrow.dataset as ds
from typing import Dict, List, Tuple, Optional, Any, Iterator
from datetime import datetime, timedelta
import logging
from tqdm import tqdm
import pickle
from collections import defaultdict, Counter
import gc
import psutil
import multiprocessing as mp
from pathlib import Path
import tempfile
import shutil
import argparse

# Optional high-performance engine
try:
    import polars as pl
except Exception:  # pragma: no cover
    pl = None

from config import data_config, token_config, model_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OMOPDataProcessor:
    """Process large OMOP datasets with memory optimization"""
    
    def __init__(self, data_path: str = None, engine: str = None, num_workers: int = None):
        self.vocab = {}
        self.vocab_size = 0
        self.quantile_mappings = {}
        self.age_mappings = {}
        self.time_interval_mappings = {}
        self.static_mappings = {}
        
        # Override data path if provided
        if data_path:
            self.omop_data_dir = data_path
        else:
            self.omop_data_dir = data_config.omop_data_dir
        
        # Engine selection
        if engine is None:
            self.engine = 'polars' if pl is not None else 'python'
        else:
            self.engine = engine
        
        # Workers
        self.num_workers = num_workers or max(1, (os.cpu_count() or 4) - 1)
        
        # Memory monitoring
        self.memory_limit_bytes = data_config.memory_limit_gb * 1024**3
        
        # Derived paths
        self.events_dir = os.path.join(data_config.output_dir, 'events_partitioned')
    
    # =====================
    # Fast path (Polars)
    # =====================
    def _scan_table_pl(self, table: str, ts_col: str, et_str: str, cid_col: str,
                       val_col: Optional[str] = None, unit_col: Optional[str] = None) -> 'pl.LazyFrame':
        path = os.path.join(self.omop_data_dir, table, "*.parquet")
        cols = ["person_id", ts_col, cid_col]
        if val_col:
            cols.append(val_col)
        if unit_col:
            cols.append(unit_col)
        base_scan = pl.scan_parquet(path)
        names = base_scan.collect_schema().names()
        lf = (base_scan
              .select([c for c in cols if c in names])
              .rename({ts_col: "ts", cid_col: "cid"}))
        # Ensure expected columns exist with unified names
        to_add = []
        # For non-measurement tables, always add null placeholders
        if val_col is None:
            to_add.append(pl.lit(None).cast(pl.Float64).alias('value_as_number'))
        elif 'value_as_number' not in names:
            to_add.append(pl.lit(None).cast(pl.Float64).alias('value_as_number'))
        if unit_col is None:
            to_add.append(pl.lit(None).cast(pl.Int64).alias('unit_concept_id'))
        elif 'unit_concept_id' not in names:
            to_add.append(pl.lit(None).cast(pl.Int64).alias('unit_concept_id'))
        if to_add:
            lf = lf.with_columns(to_add)
        # Add event type literal
        lf = lf.with_columns([pl.lit(et_str).alias("et")])
        # Cast dtypes consistently
        # Some columns may be missing as nulls; cast handles them
        lf = lf.with_columns([
            pl.col('person_id').cast(pl.Int64),
            pl.col('ts').cast(pl.Datetime("ns")),
            pl.col('cid').cast(pl.Int64),
            pl.col('value_as_number').cast(pl.Float64),
            pl.col('unit_concept_id').cast(pl.Int64),
            pl.col('et').cast(pl.Utf8),
        ])
        # Order columns identically across tables
        lf = lf.select(['person_id', 'ts', 'et', 'cid', 'value_as_number', 'unit_concept_id'])
        return lf
    
    def _build_events_polars(self) -> 'pl.LazyFrame':
        logger.info("[FAST] Scanning OMOP tables with Polars (lazy)...")
        cond = self._scan_table_pl("condition_occurrence", "condition_start_datetime", "condition", "condition_concept_id")
        drug = self._scan_table_pl("drug_exposure", "drug_exposure_start_datetime", "medication", "drug_concept_id")
        proc = self._scan_table_pl("procedure_occurrence", "procedure_datetime", "procedure", "procedure_concept_id")
        meas = self._scan_table_pl("measurement", "measurement_datetime", "measurement", "measurement_concept_id", "value_as_number", "unit_concept_id")
        obs  = self._scan_table_pl("observation", "observation_datetime", "observation", "observation_concept_id")
        events = pl.concat([cond, drug, proc, meas, obs], how="vertical_relaxed")
        events = (events
                  .filter(pl.col("ts").is_not_null() & pl.col("cid").is_not_null())
                  .with_columns([
                      pl.col("ts").cast(pl.Datetime("ns")),
                      (pl.col("person_id") % 1024).cast(pl.Int64).alias("pid_bucket")
                  ])
                  .select(["person_id", "ts", "et", "cid", "value_as_number", "unit_concept_id", "pid_bucket"]))
        return events
    
    def _write_events_partitioned(self, events_lazy: 'pl.LazyFrame') -> None:
        logger.info(f"[FAST] Writing partitioned events to {self.events_dir} ...")
        os.makedirs(self.events_dir, exist_ok=True)
        # Collect lazily (streaming) then write partitioned with PyArrow dataset API
        try:
            df_events = events_lazy.collect()
            total_rows = df_events.height
            logger.info(f"[FAST] Events rows: {total_rows:,}")
            tbl = df_events.to_arrow()
            import pyarrow.dataset as pds
            pds.write_dataset(
                tbl,
                base_dir=self.events_dir,
                format="parquet",
                partitioning=["pid_bucket"],
                existing_data_behavior="overwrite_or_ignore"
            )
            logger.info("[FAST] Events written (partitioned by pid_bucket)")
        except Exception as e:
            logger.error(f"Error writing partitioned events: {e}")
            raise
    
    def _load_person_static_pl(self) -> Dict[int, Dict[str, Any]]:
        path = os.path.join(self.omop_data_dir, "person", "*.parquet")
        logger.info("[FAST] Loading person demographics (Polars)...")
        cols = [
            "person_id", "gender_concept_id", "race_concept_id", "ethnicity_concept_id",
            "birth_datetime", "death_datetime"
        ]
        base_scan = pl.scan_parquet(path)
        names = base_scan.collect_schema().names()
        lf = base_scan.select([c for c in cols if c in names])
        if "birth_datetime" in names:
            lf = lf.with_columns(pl.col("birth_datetime").cast(pl.Datetime("ns")))
        df = lf.collect()
        person_data: Dict[int, Dict[str, Any]] = {}
        if df.height:
            for row in tqdm(df.iter_rows(named=True), total=df.height, desc="[FAST] Demographics", unit="row"):
                pid = int(row.get("person_id"))
                birth_dt = row.get("birth_datetime")
                birth_year = None
                if birth_dt is not None:
                    try:
                        birth_year = birth_dt.year
                    except Exception:
                        birth_year = None
                person_data[pid] = {
                    'gender_concept_id': row.get('gender_concept_id'),
                    'race_concept_id': row.get('race_concept_id'),
                    'ethnicity_concept_id': row.get('ethnicity_concept_id'),
                    'birth_datetime': birth_dt,
                    'death_datetime': row.get('death_datetime'),
                    'birth_year': birth_year,
                }
        logger.info(f"[FAST] Loaded demographics for {len(person_data)} patients")
        return person_data
    
    def _compute_stats_polars(self, events_lazy: 'pl.LazyFrame') -> Tuple[Counter, Dict[str, np.ndarray]]:
        logger.info("[FAST] Computing concept frequencies (Polars)...")
        counts = (events_lazy
                  .group_by(["et", "cid"])  # polars uses group_by in recent versions
                  .agg(pl.len().alias("n"))
                  .collect())
        concept_counts: Counter = Counter()
        for et, cid, n in counts.iter_rows():
            if et in ("condition", "medication", "procedure", "measurement", "observation"):
                prefix = {
                    "condition": "CONDITION_",
                    "medication": "DRUG_",
                    "procedure": "PROCEDURE_",
                    "measurement": "MEASUREMENT_",
                    "observation": "OBSERVATION_",
                }[et]
                concept_counts[f"{prefix}{cid}"] += int(n)
        
        logger.info("[FAST] Computing measurement quantiles (Polars)...")
        meas = events_lazy.filter(pl.col("et") == "measurement").select(["cid", "value_as_number"]).drop_nulls()
        # Approximate deciles per measurement concept
        # Collect to eager with streaming; then groupby in Polars
        mdf = meas.collect()
        quantile_mappings: Dict[str, np.ndarray] = {}
        if mdf.height:
            gb = mdf.group_by("cid")
            for cid, sub in gb:
                vals = sub.get_column("value_as_number").to_numpy()
                if vals.size >= 10:
                    try:
                        qs = np.percentile(vals, np.arange(0, 100, 10))
                        quantile_mappings[str(cid)] = qs
                    except Exception:
                        continue
        return concept_counts, quantile_mappings
    
    def _tokenize_from_partitions(self, person_data: Dict[int, Dict[str, Any]]) -> Dict[int, List[int]]:
        logger.info("[FAST] Tokenizing per-person partitions (sequential)...")
        tokenized: Dict[int, List[int]] = {}
        if not os.path.exists(self.events_dir):
            logger.error(f"Partitioned events directory not found: {self.events_dir}")
            return tokenized
        # Iterate over pid_bucket partitions and group by person_id inside each file
        files = list(Path(self.events_dir).rglob('*.parquet'))
        pbar_files = tqdm(files, desc="[FAST] Tokenizing partitions", unit="file")
        pbar_patients = tqdm(total=0, desc="[FAST] Patients tokenized", unit="pt")
        for file in pbar_files:
            try:
                df = pl.read_parquet(str(file))
                if df.is_empty():
                    continue
                # Ensure expected columns exist
                expected_cols = {"person_id", "ts", "et", "cid", "value_as_number", "unit_concept_id"}
                missing = expected_cols - set(df.columns)
                if missing:
                    logger.warning(f"Skipping {file} due to missing columns: {missing}")
                    continue
                # Sort for deterministic order and group by person_id
                df = df.sort(["person_id", "ts"]) 
                # Update patient total for ETA
                try:
                    num_groups = int(df.get_column("person_id").n_unique())
                    pbar_patients.total += num_groups
                    pbar_patients.refresh()
                except Exception:
                    num_groups = None
                gb = df.group_by("person_id")
                for (pid,), sub in gb:
                    pid = int(pid)
                    if pid not in person_data:
                        continue
                    sub = sub.sort("ts")
                    # Build timeline
                    pinfo = person_data[pid]
                    timeline: List[Dict[str, Any]] = [{
                        'timestamp': pinfo.get('birth_datetime', datetime.now()),
                        'event_type': 'static',
                        'gender': pinfo.get('gender_concept_id'),
                        'race': pinfo.get('race_concept_id'),
                        'ethnicity': pinfo.get('ethnicity_concept_id'),
                        'birth_year': pinfo.get('birth_year'),
                        'data': pinfo,
                    }]
                    for row in sub.iter_rows(named=True):
                        ts = row.get('ts')
                        if ts is None:
                            continue
                        et = row.get('et')
                        cid = row.get('cid')
                        val = row.get('value_as_number')
                        unit = row.get('unit_concept_id')
                        ev: Dict[str, Any] = {'timestamp': ts, 'event_type': et}
                        if et == 'condition':
                            ev['condition_concept_id'] = cid
                        elif et == 'medication':
                            ev['drug_concept_id'] = cid
                        elif et == 'procedure':
                            ev['procedure_concept_id'] = cid
                        elif et == 'measurement':
                            ev['measurement_concept_id'] = cid
                            ev['value_as_number'] = val
                            ev['unit_concept_id'] = unit
                        elif et == 'observation':
                            ev['observation_concept_id'] = cid
                        timeline.append(ev)
                    # Tokenize
                    birth_year = pinfo.get('birth_year')
                    current_year = datetime.now().year
                    patient_age = (current_year - birth_year) if birth_year else 50
                    tokens = self.tokenize_timeline(timeline, patient_age)
                    tokenized[pid] = tokens
                    pbar_patients.update(1)
            except Exception as e:
                logger.warning(f"Tokenization failed for {file}: {e}")
                continue
        pbar_files.close()
        pbar_patients.close()
        logger.info(f"[FAST] Tokenized {len(tokenized)} patients")
        return tokenized
    
    def process_all_data_fast_polars(self) -> Tuple[Dict[int, List[int]], Dict[str, int]]:
        logger.info("[FAST] Processing OMOP data with Polars/PyArrow pipeline...")
        events = self._build_events_polars()
        self._write_events_partitioned(events)
        person_data = self._load_person_static_pl()
        # Create mappings
        concept_counts, quantiles = self._compute_stats_polars(events)
        self.quantile_mappings = quantiles
        self.create_age_mappings()
        self.create_time_interval_mappings()
        # Build vocabulary using counts
        logger.info("[FAST] Building vocabulary from concept frequencies...")
        # Base tokens
        self.vocab[token_config.pad_token] = 0
        self.vocab[token_config.unk_token] = 1
        self.vocab[token_config.eos_token] = 2
        self.vocab[token_config.sos_token] = 3
        for et in ["admission","discharge","condition","medication","procedure","measurement","observation","death"]:
            self.vocab[f"EVENT_{et.upper()}"] = len(self.vocab)
        for age_interval in self.age_mappings.keys():
            self.vocab[f"AGE_{age_interval}"] = len(self.vocab)
        for time_interval in self.time_interval_mappings.keys():
            self.vocab[f"TIME_{time_interval}"] = len(self.vocab)
        for i in range(model_config.max_quantile_tokens):
            self.vocab[f"Q{i}"] = len(self.vocab)
        # Demographic tokens (from person_data)
        genders = set()
        races = set()
        year_intervals = set()
        for p in person_data.values():
            g = p.get('gender_concept_id')
            r = p.get('race_concept_id')
            by = p.get('birth_year')
            if g is not None:
                genders.add(str(g))
            if r is not None:
                races.add(str(r))
            if by is not None:
                yi = self._get_year_interval(by)
                year_intervals.add(yi)
        for g in sorted(genders):
            self.vocab[f"GENDER_{g}"] = len(self.vocab)
        for r in sorted(races):
            self.vocab[f"RACE_{r}"] = len(self.vocab)
        for yi in sorted(year_intervals):
            self.vocab[f"YEAR_{yi}"] = len(self.vocab)
        # Add concept tokens up to capacity
        max_concepts = model_config.vocab_size - len(self.vocab)
        for concept, _n in concept_counts.most_common(max_concepts):
            self.vocab[concept] = len(self.vocab)
        self.vocab_size = len(self.vocab)
        logger.info(f"[FAST] Vocabulary size: {self.vocab_size}")
        # Tokenize from partitions
        tokenized_timelines = self._tokenize_from_partitions(person_data)
        # Save
        self._save_processed_data(tokenized_timelines, {})
        return tokenized_timelines, self.vocab

    # =====================
    # Existing slower path (Python/Pandas)
    # =====================
    def get_memory_usage(self) -> float:
        """Get current memory usage in GB"""
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024**3

    def check_memory_limit(self) -> bool:
        """Check if we're approaching memory limit"""
        current_memory = self.get_memory_usage()
        return current_memory < self.memory_limit_bytes * 0.8

    def load_omop_data_chunked(self, table_name: str) -> Iterator[pd.DataFrame]:
        """Load OMOP table data in chunks to manage memory"""
        table_dir = os.path.join(self.omop_data_dir, table_name)
        
        if not os.path.exists(table_dir):
            logger.warning(f"Table directory {table_dir} does not exist")
            return
        
        # Get all parquet files for this table
        parquet_files = []
        for file in os.listdir(table_dir):
            if file.endswith('.parquet'):
                parquet_files.append(os.path.join(table_dir, file))
        
        if not parquet_files:
            logger.warning(f"No parquet files found in {table_dir}")
            logger.info(f"Contents of {table_dir}: {os.listdir(table_dir)}")
            return
        
        logger.info(f"Loading {table_name} from {len(parquet_files)} parquet files")
        logger.info(f"Files: {[os.path.basename(f) for f in parquet_files]}")
        
        # Process files in chunks
        for i in range(0, len(parquet_files), data_config.chunk_size):
            chunk_files = parquet_files[i:i + data_config.chunk_size]
            
            # Load chunk
            chunk_data = []
            for file_path in chunk_files:
                try:
                    logger.info(f"Loading file: {os.path.basename(file_path)}")
                    
                    # Try different loading methods for All of Us data
                    df = None
                    
                    # Method 1: Try custom dbdate handler first
                    try:
                        df = self._load_with_dbdate_handler(file_path)
                        if df is not None:
                            logger.info(f"Successfully loaded with custom dbdate handler")
                    except Exception as e1:
                        logger.info(f"Custom dbdate handler failed: {e1}")
                        
                        # Method 2: Try pandas read_parquet (handles custom types better)
                        try:
                            df = pd.read_parquet(file_path, engine='pyarrow')
                            logger.info(f"Successfully loaded with pandas read_parquet")
                        except Exception as e2:
                            logger.info(f"pandas read_parquet failed: {e2}")
                            
                            # Method 3: Try pyarrow with specific options
                            try:
                                import pyarrow.parquet as pq
                                table = pq.read_table(file_path, use_threads=True)
                                df = table.to_pandas()
                                logger.info(f"Successfully loaded with pyarrow read_table")
                            except Exception as e3:
                                logger.info(f"pyarrow read_table failed: {e3}")
                                
                                # Method 4: Try with fastparquet engine
                                try:
                                    df = pd.read_parquet(file_path, engine='fastparquet')
                                    logger.info(f"Successfully loaded with fastparquet engine")
                                except Exception as e4:
                                    logger.error(f"All loading methods failed for {file_path}")
                                    logger.error(f"  custom dbdate: {e1}")
                                    logger.error(f"  pandas: {e2}")
                                    logger.error(f"  pyarrow: {e3}")
                                    logger.error(f"  fastparquet: {e4}")
                                    continue
                    
                    if df is not None:
                        logger.info(f"Loaded {len(df)} rows from {os.path.basename(file_path)}")
                        logger.info(f"Columns: {list(df.columns)}")
                        logger.info(f"Data types: {df.dtypes.to_dict()}")
                        
                        # Handle All of Us specific data type issues
                        df = self._fix_all_of_us_data_types(df, table_name)
                        
                        chunk_data.append(df)
                        
                        # Check memory usage
                        if not self.check_memory_limit():
                            logger.info(f"Memory limit approaching, processing chunk {i//data_config.chunk_size + 1}")
                            break
                        
                except Exception as e:
                    logger.warning(f"Error loading {file_path}: {e}")
                    continue
            
            if chunk_data:
                # Combine chunk data
                combined_chunk = pd.concat(chunk_data, ignore_index=True)
                logger.info(f"Loaded chunk {i//data_config.chunk_size + 1}: {len(combined_chunk)} rows")
                
                yield combined_chunk
                
                # Clear memory
                del combined_chunk, chunk_data
                gc.collect()

    def _load_with_dbdate_handler(self, file_path: str) -> pd.DataFrame:
        """Custom loader for All of Us dbdate format"""
        try:
            import pyarrow.parquet as pq
            import pyarrow as pa
            
            # Try to read the data directly first
            try:
                table = pq.read_table(file_path, use_threads=True)
                df = table.to_pandas()
                logger.info(f"Successfully loaded table with {len(df)} rows")
            except Exception as e:
                logger.info(f"Direct loading failed: {e}")
                
                # If direct loading fails due to dbdate, try more aggressive approach
                if 'dbdate' in str(e).lower():
                    logger.info("Detected dbdate error, trying aggressive loading approach...")
                    try:
                        # Try to read with specific options that might handle custom types
                        df = self._load_with_aggressive_dbdate_handling(file_path)
                        if df is not None:
                            logger.info("Successfully loaded with aggressive dbdate handling")
                            return df
                    except Exception as e2:
                        logger.info(f"Aggressive dbdate handling failed: {e2}")
                
                return None
            
            # Look for columns that might be dbdate (datetime-like columns with issues)
            potential_dbdate_cols = []
            for col in df.columns:
                if df[col].dtype == 'object':
                    # Check if this looks like a date column
                    if any(keyword in col.lower() for keyword in ['date', 'time', 'start', 'end']):
                        potential_dbdate_cols.append(col)
            
            if not potential_dbdate_cols:
                logger.info("No potential dbdate columns found")
                return df
            
            logger.info(f"Found potential dbdate columns: {potential_dbdate_cols}")
            
            # Try to convert these columns to datetime
            for col in potential_dbdate_cols:
                if col in df.columns:
                    logger.info(f"Attempting to convert column: {col}")
                    
                    # Sample a few values to see what we're working with
                    sample_values = df[col].dropna().head(5).tolist()
                    logger.info(f"Sample values from {col}: {sample_values}")
                    
                    # Try different conversion strategies
                    conversion_success = False
                    
                    # Strategy 1: Try as integer days since epoch
                    try:
                        temp_df = df.copy()
                        temp_df[col] = pd.to_datetime(temp_df[col], unit='D', errors='coerce')
                        if temp_df[col].notna().sum() > 0:
                            df[col] = temp_df[col]
                            logger.info(f"Successfully converted {col} from days since epoch")
                            conversion_success = True
                    except Exception as e:
                        logger.info(f"Days since epoch conversion failed for {col}: {e}")
                    
                    # Strategy 2: Try as integer seconds since epoch
                    if not conversion_success:
                        try:
                            temp_df = df.copy()
                            temp_df[col] = pd.to_datetime(temp_df[col], unit='s', errors='coerce')
                            if temp_df[col].notna().sum() > 0:
                                df[col] = temp_df[col]
                                logger.info(f"Successfully converted {col} from seconds since epoch")
                                conversion_success = True
                        except Exception as e:
                            logger.info(f"Seconds since epoch conversion failed for {col}: {e}")
                    
                    # Strategy 3: Try as integer milliseconds since epoch
                    if not conversion_success:
                        try:
                            temp_df = df.copy()
                            temp_df[col] = pd.to_datetime(temp_df[col], unit='ms', errors='coerce')
                            if temp_df[col].notna().sum() > 0:
                                df[col] = temp_df[col]
                                logger.info(f"Successfully converted {col} from milliseconds since epoch")
                                conversion_success = True
                        except Exception as e:
                            logger.info(f"Milliseconds since epoch conversion failed for {col}: {e}")
                    
                    # Strategy 4: Try string parsing
                    if not conversion_success:
                        try:
                            temp_df = df.copy()
                            temp_df[col] = pd.to_datetime(temp_df[col], errors='coerce')
                            if temp_df[col].notna().sum() > 0:
                                df[col] = temp_df[col]
                                logger.info(f"Successfully converted {col} from string parsing")
                                conversion_success = True
                        except Exception as e:
                            logger.info(f"String parsing conversion failed for {col}: {e}")
                    
                    if not conversion_success:
                        logger.warning(f"Could not convert {col} to datetime, keeping as object")
            
            return df
            
        except Exception as e:
            logger.error(f"Error in custom dbdate handler: {e}")
            return None
    
    def _load_with_aggressive_dbdate_handling(self, file_path: str) -> pd.DataFrame:
        """More aggressive approach to handle dbdate types"""
        try:
            import pyarrow.parquet as pq
            import pyarrow as pa
            import numpy as np
            
            logger.info("Attempting aggressive dbdate handling...")
            
            # Try to read with different PyArrow options
            try:
                # Option 1: Read with specific column selection
                table = pq.read_table(file_path, use_threads=True, memory_map=True)
                df = table.to_pandas()
                logger.info(f"Aggressive loading succeeded with {len(df)} rows")
                return df
            except Exception as e1:
                logger.info(f"Option 1 failed: {e1}")
                
                try:
                    # Option 2: Try reading as raw bytes and converting
                    logger.info("Trying raw bytes approach...")
                    
                    # Read the file metadata first
                    parquet_file = pq.ParquetFile(file_path)
                    
                    # Get column names
                    column_names = [field.name for field in parquet_file.schema]
                    logger.info(f"Available columns: {column_names}")
                    
                    # Try to read specific columns that don't have dbdate
                    safe_columns = []
                    for col in column_names:
                        if any(keyword in col.lower() for keyword in ['id', 'concept', 'value', 'count', 'type']):
                            safe_columns.append(col)
                    
                    if safe_columns:
                        logger.info(f"Attempting to read safe columns: {safe_columns}")
                        table = pq.read_table(file_path, columns=safe_columns, use_threads=True)
                        df = table.to_pandas()
                        logger.info(f"Safe column loading succeeded with {len(df)} rows")
                        return df
                    else:
                        logger.warning("No safe columns found for selective loading")
                        
                except Exception as e2:
                    logger.info(f"Option 2 failed: {e2}")
                
                try:
                    # Option 3: Try with different PyArrow version compatibility
                    logger.info("Trying version compatibility approach...")
                    
                    # Force PyArrow to ignore schema issues
                    table = pq.read_table(file_path, use_threads=True, memory_map=True)
                    df = table.to_pandas()
                    logger.info(f"Version compatibility approach succeeded with {len(df)} rows")
                    return df
                    
                except Exception as e3:
                    logger.info(f"Option 3 failed: {e3}")
                
                try:
                    # Option 4: Try reading with pandas and pyarrow fallback
                    logger.info("Trying pandas with pyarrow fallback...")
                    
                    # Try pandas first
                    df = pd.read_parquet(file_path, engine='pyarrow')
                    logger.info(f"Pandas with pyarrow succeeded with {len(df)} rows")
                    return df
                    
                except Exception as e4:
                    logger.info(f"Option 4 failed: {e4}")
                
                try:
                    # Option 5: Try reading with fastparquet engine
                    logger.info("Trying fastparquet engine...")
                    
                    df = pd.read_parquet(file_path, engine='fastparquet')
                    logger.info(f"Fastparquet engine succeeded with {len(df)} rows")
                    return df
                    
                except Exception as e5:
                    logger.info(f"Option 5 failed: {e5}")
                
                try:
                    # Option 6: Try reading with pyarrow and ignore schema
                    logger.info("Trying pyarrow with schema override...")
                    
                    # Read the file and try to handle dbdate columns manually
                    table = pq.read_table(file_path, use_threads=True)
                    
                    # Convert to pandas and handle dbdate columns
                    df = table.to_pandas()
                    
                    # Look for columns that might be dbdate and convert them
                    for col in df.columns:
                        if df[col].dtype == 'object':
                            if any(keyword in col.lower() for keyword in ['date', 'time', 'start', 'end']):
                                logger.info(f"Converting potential dbdate column: {col}")
                                try:
                                    # Try to convert as integer days since epoch
                                    df[col] = pd.to_datetime(df[col], unit='D', errors='coerce')
                                    logger.info(f"Converted {col} from days since epoch")
                                except:
                                    try:
                                        # Try to convert as integer seconds since epoch
                                        df[col] = pd.to_datetime(df[col], unit='s', errors='coerce')
                                        logger.info(f"Converted {col} from seconds since epoch")
                                    except:
                                        try:
                                            # Try to convert as integer milliseconds since epoch
                                            df[col] = pd.to_datetime(df[col], unit='ms', errors='coerce')
                                            logger.info(f"Converted {col} from milliseconds since epoch")
                                        except:
                                            # Last resort: try to parse as string
                                            logger.warning(f"Could not convert {col} from numeric format, trying string parsing")
                                            df[col] = pd.to_datetime(df[col], errors='coerce')
                    
                    logger.info(f"Schema override approach succeeded with {len(df)} rows")
                    return df
                    
                except Exception as e6:
                    logger.info(f"Option 6 failed: {e6}")
            
            logger.error("All aggressive dbdate handling approaches failed")
            return None
            
        except Exception as e:
            logger.error(f"Error in aggressive dbdate handling: {e}")
            return None
    
    def _fix_all_of_us_data_types(self, df: pd.DataFrame, table_name: str) -> pd.DataFrame:
        """Fix All of Us specific data type issues"""
        logger.info(f"Fixing data types for {table_name} table")
        
        # Columns that should NOT be converted (keep as strings)
        preserve_as_strings = [
            'person_source_value', 'gender_source_value', 'race_source_value', 
            'ethnicity_source_value', 'state_of_residence_source_value', 
            'sex_at_birth_source_value', 'self_reported_category_source_value',
            'visit_source_value', 'condition_source_value', 'drug_source_value',
            'procedure_source_value', 'measurement_source_value', 'observation_source_value',
            'death_source_value', 'note_source_value', 'specimen_source_value'
        ]
        
        # Convert problematic columns
        for col in df.columns:
            if df[col].dtype == 'object':
                # Skip columns that should remain as strings
                if col in preserve_as_strings:
                    logger.info(f"Preserving {col} as string (source value column)")
                    continue
                
                # Try to convert to datetime if it looks like a date
                if any(keyword in col.lower() for keyword in ['date', 'time', 'start', 'end']):
                    try:
                        # Handle All of Us custom date format
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        logger.info(f"Converted {col} to datetime")
                    except Exception as e:
                        logger.info(f"Could not convert {col} to datetime: {e}")
                
                # Try to convert to numeric if it looks like a number (but not source values)
                elif any(keyword in col.lower() for keyword in ['id', 'concept', 'value', 'count']) and 'source' not in col.lower():
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        logger.info(f"Converted {col} to numeric")
                    except Exception as e:
                        logger.info(f"Could not convert {col} to numeric: {e}")
        
        # Handle specific table issues
        if table_name == 'person':
            # Fix birth_datetime if it's not already a datetime
            if 'birth_datetime' in df.columns and df['birth_datetime'].dtype != 'datetime64[ns]':
                try:
                    df['birth_datetime'] = pd.to_datetime(df['birth_datetime'], errors='coerce')
                    logger.info("Fixed birth_datetime column")
                except Exception as e:
                    logger.warning(f"Could not fix birth_datetime: {e}")
        
        elif table_name in ['visit_occurrence', 'condition_occurrence', 'drug_exposure', 
                           'procedure_occurrence', 'measurement', 'observation']:
            # Fix datetime columns
            datetime_cols = [col for col in df.columns if any(keyword in col.lower() 
                           for keyword in ['date', 'time', 'start', 'end'])]
            
            for col in datetime_cols:
                if col in df.columns and df[col].dtype != 'datetime64[ns]':
                    try:
                        df[col] = pd.to_datetime(df[col], errors='coerce')
                        logger.info(f"Fixed {col} column")
                    except Exception as e:
                        logger.warning(f"Could not fix {col}: {e}")
        
        logger.info(f"Data type fixes completed for {table_name}")
        return df
    
    def process_person_table(self) -> Dict[int, Dict]:
        """Process person table to get patient demographics"""
        logger.info("Processing person table...")
        
        person_data = {}
        
        for chunk in self.load_omop_data_chunked("person"):
            logger.info(f"Processing person chunk with {len(chunk)} rows")
            logger.info(f"Person chunk columns: {list(chunk.columns)}")
            logger.info(f"First few person IDs: {chunk['person_id'].head().tolist() if 'person_id' in chunk.columns else 'No person_id column'}")
            
            for _, person in chunk.iterrows():
                person_id = person['person_id']
                person_data[person_id] = {
                    'gender_concept_id': person.get('gender_concept_id'),
                    'race_concept_id': person.get('race_concept_id'),
                    'ethnicity_concept_id': person.get('ethnicity_concept_id'),
                    'birth_datetime': person.get('birth_datetime'),
                    'death_datetime': person.get('death_datetime'),
                    'birth_year': person.get('birth_datetime', datetime.now()).year if pd.notna(person.get('birth_datetime')) else None
                }
        
        logger.info(f"Processed {len(person_data)} patients")
        if person_data:
            logger.info(f"Sample person data: {list(person_data.items())[:3]}")
        return person_data
    
    def process_visit_occurrence(self, person_data: Dict[int, Dict]) -> Dict[int, List[Dict]]:
        """Process visit occurrence table to get admission/discharge events"""
        logger.info("Processing visit occurrence table...")
        
        visit_timelines = defaultdict(list)
        
        for chunk in self.load_omop_data_chunked("visit_occurrence"):
            for _, visit in chunk.iterrows():
                patient_id = visit['person_id']
                
                if patient_id not in person_data:
                    continue
                
                # Admission event
                if pd.notna(visit['visit_start_datetime']):
                    visit_timelines[patient_id].append({
                        'timestamp': visit['visit_start_datetime'],
                        'event_type': 'admission',
                        'visit_id': visit['visit_occurrence_id'],
                        'visit_type': visit.get('visit_type_concept_id', 'unknown'),
                        'data': visit.to_dict()
                    })
                
                # Discharge event
                if pd.notna(visit['visit_end_datetime']):
                    visit_timelines[patient_id].append({
                        'timestamp': visit['visit_end_datetime'],
                        'event_type': 'discharge',
                        'visit_id': visit['visit_occurrence_id'],
                        'data': visit.to_dict()
                    })
        
        logger.info(f"Processed visits for {len(visit_timelines)} patients")
        return dict(visit_timelines)
    
    def process_condition_occurrence(self, person_data: Dict[int, Dict]) -> Dict[int, List[Dict]]:
        """Process condition occurrence table"""
        logger.info("Processing condition occurrence table...")
        
        condition_timelines = defaultdict(list)
        
        for chunk in self.load_omop_data_chunked("condition_occurrence"):
            for _, condition in chunk.iterrows():
                patient_id = condition['person_id']
                
                if patient_id not in person_data:
                    continue
                    
                if pd.notna(condition['condition_start_datetime']):
                    condition_timelines[patient_id].append({
                        'timestamp': condition['condition_start_datetime'],
                        'event_type': 'condition',
                        'condition_concept_id': condition['condition_concept_id'],
                        'data': condition.to_dict()
                    })
        
        logger.info(f"Processed conditions for {len(condition_timelines)} patients")
        return dict(condition_timelines)
    
    def process_drug_exposure(self, person_data: Dict[int, Dict]) -> Dict[int, List[Dict]]:
        """Process drug exposure table"""
        logger.info("Processing drug exposure table...")
        
        drug_timelines = defaultdict(list)
        
        for chunk in self.load_omop_data_chunked("drug_exposure"):
            for _, drug in chunk.iterrows():
                patient_id = drug['person_id']
                
                if patient_id not in person_data:
                    continue
                    
                if pd.notna(drug['drug_exposure_start_datetime']):
                    drug_timelines[patient_id].append({
                        'timestamp': drug['drug_exposure_start_datetime'],
                        'event_type': 'medication',
                        'drug_concept_id': drug['drug_concept_id'],
                        'data': drug.to_dict()
                    })
        
        logger.info(f"Processed drug exposures for {len(drug_timelines)} patients")
        return dict(drug_timelines)
    
    def process_procedure_occurrence(self, person_data: Dict[int, Dict]) -> Dict[int, List[Dict]]:
        """Process procedure occurrence table"""
        logger.info("Processing procedure occurrence table...")
        
        procedure_timelines = defaultdict(list)
        
        for chunk in self.load_omop_data_chunked("procedure_occurrence"):
            for _, procedure in chunk.iterrows():
                patient_id = procedure['person_id']
                
                if patient_id not in person_data:
                    continue
                    
                if pd.notna(procedure['procedure_datetime']):
                    procedure_timelines[patient_id].append({
                        'timestamp': procedure['procedure_datetime'],
                        'event_type': 'procedure',
                        'procedure_concept_id': procedure['procedure_concept_id'],
                        'data': procedure.to_dict()
                    })
        
        logger.info(f"Processed procedures for {len(procedure_timelines)} patients")
        return dict(procedure_timelines)
    
    def process_measurement(self, person_data: Dict[int, Dict]) -> Dict[int, List[Dict]]:
        """Process measurement table"""
        logger.info("Processing measurement table...")
        
        measurement_timelines = defaultdict(list)
        
        for chunk in self.load_omop_data_chunked("measurement"):
            for _, measurement in chunk.iterrows():
                patient_id = measurement['person_id']
                
                if patient_id not in person_data:
                    continue
                    
                if pd.notna(measurement['measurement_datetime']):
                    measurement_timelines[patient_id].append({
                        'timestamp': measurement['measurement_datetime'],
                        'event_type': 'measurement',
                        'measurement_concept_id': measurement['measurement_concept_id'],
                        'value_as_number': measurement.get('value_as_number'),
                        'unit_concept_id': measurement.get('unit_concept_id'),
                        'data': measurement.to_dict()
                    })
        
        logger.info(f"Processed measurements for {len(measurement_timelines)} patients")
        return dict(measurement_timelines)
    
    def process_observation(self, person_data: Dict[int, Dict]) -> Dict[int, List[Dict]]:
        """Process observation table"""
        logger.info("Processing observation table...")
        
        observation_timelines = defaultdict(list)
        
        for chunk in self.load_omop_data_chunked("observation"):
            for _, observation in chunk.iterrows():
                patient_id = observation['person_id']
                
                if patient_id not in person_data:
                    continue
                    
                if pd.notna(observation['observation_datetime']):
                    observation_timelines[patient_id].append({
                        'timestamp': observation['observation_datetime'],
                        'event_type': 'observation',
                        'observation_concept_id': observation['observation_concept_id'],
                        'data': observation.to_dict()
                    })
        
        logger.info(f"Processed observations for {len(observation_timelines)} patients")
        return dict(observation_timelines)
    
    def process_death(self, person_data: Dict[int, Dict]) -> Dict[int, List[Dict]]:
        """Process death table"""
        logger.info("Processing death table...")
        
        death_timelines = defaultdict(list)
        
        for chunk in self.load_omop_data_chunked("death"):
            for _, death in chunk.iterrows():
                patient_id = death['person_id']
                
                if patient_id not in person_data:
                    continue
                    
                if pd.notna(death['death_datetime']):
                    death_timelines[patient_id].append({
                        'timestamp': death['death_datetime'],
                        'event_type': 'death',
                        'data': death.to_dict()
                    })
        
        logger.info(f"Processed deaths for {len(death_timelines)} patients")
        return dict(death_timelines)
    
    def merge_patient_timelines(self, person_data: Dict[int, Dict], 
                               *timeline_dicts) -> Dict[int, List[Dict]]:
        """Merge all timeline dictionaries into unified patient timelines"""
        logger.info("Merging patient timelines...")
        
        patient_timelines = defaultdict(list)
        
        # Add static information first
        for patient_id, person_info in person_data.items():
            patient_timelines[patient_id].append({
                'timestamp': person_info.get('birth_datetime', datetime.now()),
                'event_type': 'static',
                'gender': person_info.get('gender_concept_id'),
                'race': person_info.get('race_concept_id'),
                'ethnicity': person_info.get('ethnicity_concept_id'),
                'birth_year': person_info.get('birth_year'),
                'data': person_info
            })
        
        # Merge all event timelines
        for timeline_dict in timeline_dicts:
            for patient_id, events in timeline_dict.items():
                if patient_id in person_data:  # Only include patients with demographics
                    patient_timelines[patient_id].extend(events)
        
        # Sort events by timestamp for each patient
        for patient_id in patient_timelines:
            patient_timelines[patient_id].sort(key=lambda x: x['timestamp'])
        
        logger.info(f"Merged timelines for {len(patient_timelines)} patients")
        return dict(patient_timelines)
    
    def create_quantile_mappings(self, patient_timelines: Dict[int, List[Dict]]):
        """Create quantile mappings for numerical values with memory optimization"""
        logger.info("Creating quantile mappings...")
        
        # Collect numerical values by concept in chunks
        numerical_values = defaultdict(list)
        
        for patient_id, timeline in patient_timelines.items():
            for event in timeline:
                if event['event_type'] == 'measurement' and 'value_as_number' in event:
                    value = event['value_as_number']
                    if pd.notna(value) and isinstance(value, (int, float)):
                        concept_id = event.get('measurement_concept_id', 'unknown')
                        numerical_values[concept_id].append(value)
                        
                        # Check memory usage
                        if not self.check_memory_limit():
                            logger.info("Memory limit approaching during quantile creation")
                            break
            
            # Process in smaller batches
            if len(numerical_values) > 1000:  # Limit number of concepts in memory
                self._process_quantile_batch(numerical_values)
                numerical_values.clear()
                gc.collect()
        
        # Process final batch
        if numerical_values:
            self._process_quantile_batch(numerical_values)
        
        logger.info(f"Created quantile mappings for {len(self.quantile_mappings)} measurement concepts")
    
    def _process_quantile_batch(self, numerical_values: Dict[str, List[float]]):
        """Process a batch of numerical values for quantile creation"""
        for concept_id, values in numerical_values.items():
            if len(values) >= 10:  # Need at least 10 values for 10 quantiles
                try:
                    quantiles = np.percentile(values, np.arange(0, 100, 10))
                    self.quantile_mappings[concept_id] = quantiles
                except Exception as e:
                    logger.warning(f"Error creating quantiles for concept {concept_id}: {e}")
    
    def create_age_mappings(self):
        """Create age interval mappings"""
        # Age intervals: 0-5, 5-10, 10-15, ..., 95-100, 100+
        age_ranges = []
        for i in range(0, 100, 5):
            age_ranges.append((i, i + 5))
        age_ranges.append((100, 120))  # 100+ years
        
        for i, (start, end) in enumerate(age_ranges):
            age_interval = f"{start}-{end}"
            self.age_mappings[age_interval] = i
        
        logger.info(f"Created {len(self.age_mappings)} age interval mappings")
        logger.info(f"Age intervals: {list(self.age_mappings.keys())[:5]}...")  # Show first 5 for debugging
    
    def create_time_interval_mappings(self):
        """Create time interval mappings"""
        # Time intervals: 5m, 15m, 1h, 3h, 6h, 12h, 1d, 3d, 1w, 1m, 3m, 6m, 1y
        intervals = [
            (5, "5m"), (15, "15m"), (60, "1h"), (180, "3h"), (360, "6h"),
            (720, "12h"), (1440, "1d"), (4320, "3d"), (10080, "1w"),
            (43200, "1m"), (129600, "3m"), (259200, "6m"), (525600, "1y")
        ]
        
        for i, (minutes, label) in enumerate(intervals):
            self.time_interval_mappings[label] = i
        
        logger.info(f"Created {len(self.time_interval_mappings)} time interval mappings")
    
    def build_vocabulary(self, patient_timelines: Dict[int, List[Dict]]):
        """Build vocabulary from patient timelines with memory optimization"""
        logger.info("Building vocabulary...")
        
        # Start with special tokens
        self.vocab[token_config.pad_token] = 0
        self.vocab[token_config.unk_token] = 1
        self.vocab[token_config.eos_token] = 2
        self.vocab[token_config.sos_token] = 3
        
        # Add event type tokens
        event_types = ["admission", "discharge", "condition", "medication", 
                      "procedure", "measurement", "observation", "death"]
        for event_type in event_types:
            self.vocab[f"EVENT_{event_type.upper()}"] = len(self.vocab)
        
        # Add age interval tokens
        for age_interval in self.age_mappings.keys():
            self.vocab[f"AGE_{age_interval}"] = len(self.vocab)
        
        # Add time interval tokens
        for time_interval in self.time_interval_mappings.keys():
            self.vocab[f"TIME_{time_interval}"] = len(self.vocab)
        
        # Add quantile tokens
        for i in range(model_config.max_quantile_tokens):
            self.vocab[f"Q{i}"] = len(self.vocab)
        
        # Scan timelines once to collect categorical values used by tokenization
        genders = set()
        races = set()
        year_intervals = set()
        visit_types = set()
        units = set()

        for patient_id, timeline in patient_timelines.items():
            # Static event
            static_event = next((e for e in timeline if e.get('event_type') == 'static'), None)
            if static_event:
                g = static_event.get('gender')
                r = static_event.get('race')
                if g is not None:
                    genders.add(g)
                if r is not None:
                    races.add(r)
                # Birth year interval as used by _get_year_interval
                by = static_event.get('birth_year')
                if by:
                    base_year = 1970
                    interval = (by - base_year) // 5
                    start_year = base_year + interval * 5
                    end_year = start_year + 5
                    year_intervals.add(f"{start_year}-{end_year}")
            # Events
            for ev in timeline:
                if ev.get('event_type') == 'admission':
                    vt = ev.get('visit_type', 'unknown')
                    visit_types.add(vt)
                elif ev.get('event_type') == 'measurement':
                    u = ev.get('unit_concept_id', 'unknown')
                    if u is not None:
                        units.add(u)

        # Add categorical tokens used by tokenization to avoid defaulting to PAD (0)
        for g in genders:
            self.vocab[f"GENDER_{g}"] = len(self.vocab)
        for r in races:
            self.vocab[f"RACE_{r}"] = len(self.vocab)
        for yi in sorted(year_intervals):
            self.vocab[f"YEAR_{yi}"] = len(self.vocab)
        for vt in visit_types:
            self.vocab[f"VISIT_TYPE_{vt}"] = len(self.vocab)
        for u in units:
            self.vocab[f"UNIT_{u}"] = len(self.vocab)

        # Add concept tokens from data (process in chunks)
        concept_counts = Counter()
        
        # Process patients in chunks to manage memory
        patient_ids = list(patient_timelines.keys())
        chunk_size = data_config.max_patients_per_chunk
        
        for i in range(0, len(patient_ids), chunk_size):
            chunk_ids = patient_ids[i:i + chunk_size]
            
            for patient_id in chunk_ids:
                timeline = patient_timelines[patient_id]
                for event in timeline:
                    if event['event_type'] == 'condition':
                        concept_id = event.get('condition_concept_id', 'unknown')
                        concept_counts[f"CONDITION_{concept_id}"] += 1
                    elif event['event_type'] == 'medication':
                        concept_id = event.get('drug_concept_id', 'unknown')
                        concept_counts[f"DRUG_{concept_id}"] += 1
                    elif event['event_type'] == 'procedure':
                        concept_id = event.get('procedure_concept_id', 'unknown')
                        concept_counts[f"PROCEDURE_{concept_id}"] += 1
                    elif event['event_type'] == 'measurement':
                        concept_id = event.get('measurement_concept_id', 'unknown')
                        concept_counts[f"MEASUREMENT_{concept_id}"] += 1
                    elif event['event_type'] == 'observation':
                        concept_id = event.get('observation_concept_id', 'unknown')
                        concept_counts[f"OBSERVATION_{concept_id}"] += 1
            
            # Clear memory after each chunk
            gc.collect()
        
        # Add most frequent concepts (limit vocabulary size)
        max_concepts = model_config.vocab_size - len(self.vocab)
        logger.info(f"Available slots for concepts: {max_concepts}")
        logger.info(f"Total concepts found: {len(concept_counts)}")
        
        # Show top concepts by frequency
        top_concepts = concept_counts.most_common(10)
        logger.info(f"Top 10 concepts by frequency: {top_concepts}")
        
        for concept, count in concept_counts.most_common(max_concepts):
            self.vocab[concept] = len(self.vocab)
        
        self.vocab_size = len(self.vocab)
        logger.info(f"Built vocabulary with {self.vocab_size} tokens")
        logger.info(f"Breakdown:")
        logger.info(f"  - Special tokens: 4")
        logger.info(f"  - Event types: 8") 
        logger.info(f"  - Age intervals: {len(self.age_mappings)}")
        logger.info(f"  - Time intervals: {len(self.time_interval_mappings)}")
        logger.info(f"  - Quantile tokens: {model_config.max_quantile_tokens}")
        logger.info(f"  - Concept tokens: {len([k for k in self.vocab.keys() if k.startswith(('CONDITION_', 'DRUG_', 'PROCEDURE_', 'MEASUREMENT_', 'OBSERVATION_'))])}")
    
    def tokenize_timeline(self, timeline: List[Dict], patient_age: float) -> List[int]:
        """Convert a patient timeline to tokens"""
        tokens = []
        
        # Add static tokens at the beginning
        static_event = next((e for e in timeline if e['event_type'] == 'static'), None)
        if static_event:
            tokens.extend(self._tokenize_static_info(static_event, patient_age))
        
        # Process chronological events
        chronological_events = [e for e in timeline if e['event_type'] != 'static']
        chronological_events.sort(key=lambda x: x['timestamp'])
        
        for i, event in enumerate(chronological_events):
            # Add time interval token if needed
            if i > 0:
                time_diff = self._calculate_time_difference(
                    chronological_events[i-1]['timestamp'], 
                    event['timestamp']
                )
                if time_diff >= model_config.min_time_interval:
                    interval_tokens = self._get_time_interval_tokens(time_diff)
                    if interval_tokens:
                        tokens.extend(interval_tokens)
            
            # Add event tokens
            event_tokens = self._tokenize_event(event)
            tokens.extend(event_tokens)
        
        # Add end of sequence token
        tokens.append(self.vocab.get(token_config.eos_token, 0))
        
        return tokens
    
    def _tokenize_static_info(self, static_event: Dict, patient_age: float) -> List[int]:
        """Tokenize static patient information"""
        tokens = []
        
        # Age token
        age_interval = self._get_age_interval(patient_age)
        age_token = self.vocab.get(f"AGE_{age_interval}", 0)
        tokens.append(age_token)
        
        # Gender token
        gender = static_event.get('gender', 'unknown')
        gender_token = self.vocab.get(f"GENDER_{gender}", 0)
        tokens.append(gender_token)
        
        # Race token
        race = static_event.get('race', 'unknown')
        race_token = self.vocab.get(f"RACE_{race}", 0)
        tokens.append(race_token)
        
        # Birth year token
        birth_year = static_event.get('birth_year', 1970)
        year_interval = self._get_year_interval(birth_year)
        year_token = self.vocab.get(f"YEAR_{year_interval}", 0)
        tokens.append(year_token)
        
        return tokens
    
    def _tokenize_event(self, event: Dict) -> List[int]:
        """Tokenize a single event"""
        tokens = []
        event_type = event['event_type']
        
        # Event type token
        type_token = self.vocab.get(f"EVENT_{event_type.upper()}", 0)
        tokens.append(type_token)
        
        # Event-specific tokens
        if event_type == 'admission':
            visit_type = event.get('visit_type', 'unknown')
            tokens.append(self.vocab.get(f"VISIT_TYPE_{visit_type}", 0))
            
        elif event_type == 'condition':
            concept_id = event.get('condition_concept_id', 'unknown')
            tokens.append(self.vocab.get(f"CONDITION_{concept_id}", 0))
            
        elif event_type == 'medication':
            concept_id = event.get('drug_concept_id', 'unknown')
            tokens.append(self.vocab.get(f"DRUG_{concept_id}", 0))
            
        elif event_type == 'procedure':
            concept_id = event.get('procedure_concept_id', 'unknown')
            tokens.append(self.vocab.get(f"PROCEDURE_{concept_id}", 0))
            
        elif event_type == 'measurement':
            concept_id = event.get('measurement_concept_id', 'unknown')
            tokens.append(self.vocab.get(f"MEASUREMENT_{concept_id}", 0))
            
            # Value quantile token
            value = event.get('value_as_number')
            if pd.notna(value) and concept_id in self.quantile_mappings:
                quantile = self._get_quantile(value, concept_id)
                quantile_token = self.vocab.get(f"Q{quantile}", 0)
                tokens.append(quantile_token)
            
            # Unit token
            unit = event.get('unit_concept_id', 'unknown')
            tokens.append(self.vocab.get(f"UNIT_{unit}", 0))
        elif event_type == 'observation':
            concept_id = event.get('observation_concept_id', 'unknown')
            tokens.append(self.vocab.get(f"OBSERVATION_{concept_id}", 0))
        
        return tokens
    
    def _get_age_interval(self, age: float) -> str:
        """Get age interval string"""
        for age_interval in self.age_mappings.keys():
            try:
                # Parse age interval string like "0-5", "5-10", etc.
                if '-' in age_interval:
                    start_age, end_age = map(int, age_interval.split('-'))
                    if start_age <= age < end_age:
                        return age_interval
                else:
                    # Handle single age values if they exist
                    single_age = int(age_interval)
                    if age == single_age:
                        return age_interval
            except (ValueError, AttributeError) as e:
                logger.warning(f"Could not parse age interval '{age_interval}': {e}")
                continue
        
        # Default for very old patients
        return "100-120"
    
    def _get_year_interval(self, year: int) -> str:
        """Get year interval string (5-year intervals)"""
        base_year = 1970
        interval = (year - base_year) // 5
        start_year = base_year + interval * 5
        end_year = start_year + 5
        return f"{start_year}-{end_year}"
    
    def _get_quantile(self, value: float, concept_id: str) -> int:
        """Get quantile for a numerical value"""
        if concept_id in self.quantile_mappings:
            quantiles = self.quantile_mappings[concept_id]
            for i, q in enumerate(quantiles):
                if value <= q:
                    return i
        return 0  # Default to first quantile
    
    def _calculate_time_difference(self, time1: datetime, time2: datetime) -> float:
        """Calculate time difference in minutes"""
        if pd.isna(time1) or pd.isna(time2):
            return 0
        diff = time2 - time1
        return diff.total_seconds() / 60
    
    def _get_time_interval_tokens(self, minutes: float) -> List[int]:
        """Return time-interval token ids per ETHOS rules.
        - No token if minutes < 5m (min_time_interval)
        - If minutes > 1y, emit multiple 6m tokens, rounding count
        - Else emit single closest bucket among the 13 bins
        """
        if minutes < model_config.min_time_interval:
            return []
        six_months = 259200  # minutes
        one_year = 525600
        if minutes > one_year:
            n = max(2, int(round(minutes / six_months)))
            sixm_token = self.vocab.get("TIME_6m")
            if sixm_token is None:
                sixm_token = self.time_interval_mappings.get("6m")
            return [sixm_token] * n if sixm_token is not None else []
        buckets = [
            (one_year, "1y"),
            (259200, "6m"),
            (129600, "3m"),
            (43200, "1m"),
            (10080, "1w"),
            (4320, "3d"),
            (1440, "1d"),
            (720, "12h"),
            (360, "6h"),
            (180, "3h"),
            (60, "1h"),
            (15, "15m"),
            (5, "5m"),
        ]
        for threshold, label in buckets:
            if minutes >= threshold:
                token_id = self.vocab.get(f"TIME_{label}")
                if token_id is None:
                    token_id = self.time_interval_mappings.get(label)
                return [token_id] if token_id is not None else []
        return []
    
    def process_all_data(self) -> Tuple[Dict[int, List[int]], Dict[str, int]]:
        """Process all OMOP data and return tokenized timelines and vocabulary"""
        logger.info("Processing all OMOP data...")
        logger.info(f"Data directory: {self.omop_data_dir}")
        logger.info(f"Memory limit: {data_config.memory_limit_gb:.1f} GB")
        logger.info(f"Engine: {self.engine}")
        
        if self.engine == 'polars' and pl is not None:
            return self.process_all_data_fast_polars()
        # Fallback to existing slower pipeline
        # Step 1: Process person table (demographics)
        person_data = self.process_person_table()
        # Step 2: Process clinical tables
        visit_timelines = self.process_visit_occurrence(person_data)
        condition_timelines = self.process_condition_occurrence(person_data)
        drug_timelines = self.process_drug_exposure(person_data)
        procedure_timelines = self.process_procedure_occurrence(person_data)
        measurement_timelines = self.process_measurement(person_data)
        observation_timelines = self.process_observation(person_data)
        death_timelines = self.process_death(person_data)
        # Step 3: Merge all timelines
        patient_timelines = self.merge_patient_timelines(
            person_data, visit_timelines, condition_timelines, drug_timelines,
            procedure_timelines, measurement_timelines, observation_timelines, death_timelines
        )
        # Step 4: Create mappings
        self.create_quantile_mappings(patient_timelines)
        self.create_age_mappings()
        self.create_time_interval_mappings()
        # Step 5: Build vocabulary
        self.build_vocabulary(patient_timelines)
        # Step 6: Tokenize timelines (in chunks to manage memory)
        tokenized_timelines = {}
        patient_ids = list(patient_timelines.keys())
        chunk_size = data_config.max_patients_per_chunk
        logger.info(f"Tokenizing {len(patient_ids)} patient timelines in chunks of {chunk_size}")
        pbar = tqdm(total=len(patient_ids), desc="Tokenizing timelines", unit="pt")
        for i in range(0, len(patient_ids), chunk_size):
            chunk_ids = patient_ids[i:i + chunk_size]
            logger.info(f"Processing chunk {i//chunk_size + 1}/{(len(patient_ids) + chunk_size - 1)//chunk_size}")
            for patient_id in chunk_ids:
                timeline = patient_timelines[patient_id]
                # Calculate patient age
                static_event = next((e for e in timeline if e['event_type'] == 'static'), None)
                if static_event and 'birth_year' in static_event:
                    birth_year = static_event['birth_year']
                    if birth_year:
                        current_year = datetime.now().year
                        patient_age = current_year - birth_year
                    else:
                        patient_age = 50
                else:
                    patient_age = 50
                tokens = self.tokenize_timeline(timeline, patient_age)
                tokenized_timelines[patient_id] = tokens
                pbar.update(1)
            gc.collect()
        pbar.close()
        self._save_processed_data(tokenized_timelines, patient_timelines)
        return tokenized_timelines, self.vocab
    
    def _save_processed_data(self, tokenized_timelines: Dict[int, List[int]], 
                           patient_timelines: Dict[int, List[Dict]]):
        """Save processed data to files"""
        # Save tokenized timelines
        with open(os.path.join(data_config.output_dir, 'tokenized_timelines.pkl'), 'wb') as f:
            pickle.dump(tokenized_timelines, f)
        
        # Save vocabulary
        with open(os.path.join(data_config.output_dir, 'vocabulary.pkl'), 'wb') as f:
            pickle.dump(self.vocab, f)
        
        # Save quantile mappings
        with open(os.path.join(data_config.output_dir, 'quantile_mappings.pkl'), 'wb') as f:
            pickle.dump(self.quantile_mappings, f)
        
        # Save patient timelines (for reference)
        with open(os.path.join(data_config.output_dir, 'patient_timelines.pkl'), 'wb') as f:
            pickle.dump(patient_timelines, f)
        
        logger.info(f"Saved processed data to {data_config.output_dir}")
    
    def load_processed_data(self) -> Tuple[Dict[int, List[int]], Dict[str, int]]:
        """Load previously processed data"""
        logger.info("Loading processed data...")
        
        # Load tokenized timelines
        with open(os.path.join(data_config.output_dir, 'tokenized_timelines.pkl'), 'rb') as f:
            tokenized_timelines = pickle.load(f)
        
        # Load vocabulary
        with open(os.path.join(data_config.output_dir, 'vocabulary.pkl'), 'rb') as f:
            self.vocab = pickle.load(f)
            self.vocab_size = len(self.vocab)
        
        # Load quantile mappings
        with open(os.path.join(data_config.output_dir, 'quantile_mappings.pkl'), 'rb') as f:
            self.quantile_mappings = pickle.load(f)
        
        logger.info(f"Loaded processed data: {len(tokenized_timelines)} patients, {self.vocab_size} tokens")
        return tokenized_timelines, self.vocab

def main():
    """Main function with command line argument parsing"""
    parser = argparse.ArgumentParser(description='Process OMOP data into tokenized Patient Health Timelines')
    parser.add_argument('--data_path', type=str, default=None,
                       help='Path to OMOP data directory (default: omop_data/)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for processed data (default: processed_data/)')
    parser.add_argument('--memory_limit', type=float, default=None,
                       help='Memory limit in GB (default: 8.0)')
    parser.add_argument('--force_reprocess', action='store_true',
                       help='Force reprocessing even if data exists')
    parser.add_argument('--tag', type=str, default=None,
                       help='Dataset tag for isolating different datasets (e.g., aou_2023, mimic_iv)')
    parser.add_argument('--engine', type=str, choices=['polars','arrow','python'], default='polars',
                       help='Processing engine: polars (fast), arrow, or python (fallback)')
    parser.add_argument('--num_workers', type=int, default=None,
                       help='Number of workers for parallel steps (default: CPU cores - 1)')

    args = parser.parse_args()
    
    # Override config values if provided
    if args.output_dir:
        data_config.output_dir = args.output_dir
    elif args.tag:
        # Use tag-based output directory
        data_config.output_dir = f"processed_data_{args.tag}"
    
    if args.memory_limit:
        data_config.memory_limit_gb = args.memory_limit
    
    # Create output directory
    os.makedirs(data_config.output_dir, exist_ok=True)
    
    # Initialize processor with custom data path and engine
    processor = OMOPDataProcessor(data_path=args.data_path, engine=args.engine, num_workers=args.num_workers)
    
    # Check if we should reprocess
    should_reprocess = args.force_reprocess
    
    # Check if existing data is valid (has patients)
    if not should_reprocess and os.path.exists(os.path.join(data_config.output_dir, 'tokenized_timelines.pkl')):
        try:
            with open(os.path.join(data_config.output_dir, 'tokenized_timelines.pkl'), 'rb') as f:
                existing_data = pickle.load(f)
            if len(existing_data) == 0:
                logger.info("Existing processed data has 0 patients, forcing reprocessing...")
                should_reprocess = True
        except Exception as e:
            logger.warning(f"Error checking existing data: {e}, forcing reprocessing...")
            should_reprocess = True
    
    # Process data if needed
    if should_reprocess or not os.path.exists(os.path.join(data_config.output_dir, 'tokenized_timelines.pkl')):
        logger.info("Processing new OMOP data...")
        tokenized_timelines, vocab = processor.process_all_data()
    else:
        logger.info("Loading existing processed data...")
        tokenized_timelines, vocab = processor.load_processed_data()
    
    print(f"Processed {len(tokenized_timelines)} patient timelines")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Data saved to: {data_config.output_dir}")
    
    # Warn if still 0 patients
    if len(tokenized_timelines) == 0:
        print("\n⚠️  WARNING: No patients were processed!")
        print("This could indicate:")
        print("1. Data directory structure is incorrect")
        print("2. Parquet files are empty or corrupted")
        print("3. Column names don't match expected OMOP format")
        print("\nTry running with --force_reprocess to see detailed logs")
    
    # Show tag information
    if args.tag:
        print(f"\n📁 Dataset tag: {args.tag}")
        print(f"📁 Output directory: {data_config.output_dir}")
        print(f"💡 To use this dataset in other scripts, use: --data_dir {data_config.output_dir}")

if __name__ == "__main__":
    main()
