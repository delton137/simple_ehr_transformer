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

from config import data_config, token_config, model_config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OMOPDataProcessor:
    """Process large OMOP datasets with memory optimization"""
    
    def __init__(self, data_path: str = None):
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
        
        # Memory monitoring
        self.memory_limit_bytes = data_config.memory_limit_gb * 1024**3
        
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
                    # Read parquet file
                    table = pq.read_table(file_path)
                    df = table.to_pandas()
                    logger.info(f"Loaded {len(df)} rows from {os.path.basename(file_path)}")
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
            self.age_mappings[f"{start}-{end}"] = i
        
        logger.info(f"Created {len(self.age_mappings)} age interval mappings")
    
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
            
            # Clear memory after each chunk
            gc.collect()
        
        # Add most frequent concepts (limit vocabulary size)
        max_concepts = model_config.vocab_size - len(self.vocab)
        for concept, count in concept_counts.most_common(max_concepts):
            self.vocab[concept] = len(self.vocab)
        
        self.vocab_size = len(self.vocab)
        logger.info(f"Built vocabulary with {self.vocab_size} tokens")
    
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
                    interval_token = self._get_time_interval_token(time_diff)
                    if interval_token is not None:
                        tokens.append(interval_token)
            
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
        
        return tokens
    
    def _get_age_interval(self, age: float) -> str:
        """Get age interval string"""
        for start, end in self.age_mappings.keys():
            start_age, end_age = map(int, start.split('-'))
            if start_age <= age < end_age:
                return f"{start_age}-{end_age}"
        return "100-120"  # Default for very old patients
    
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
    
    def _get_time_interval_token(self, minutes: float) -> Optional[int]:
        """Get time interval token for given minutes"""
        if minutes < model_config.min_time_interval:
            return None
        
        # Find the closest interval
        for label, token_id in self.time_interval_mappings.items():
            if label == "1y" and minutes >= 525600:
                return token_id
            elif label == "6m" and 259200 <= minutes < 525600:
                return token_id
            elif label == "3m" and 129600 <= minutes < 259200:
                return token_id
            elif label == "1m" and 43200 <= minutes < 129600:
                return token_id
            elif label == "1w" and 10080 <= minutes < 43200:
                return token_id
            elif label == "3d" and 4320 <= minutes < 10080:
                return token_id
            elif label == "1d" and 1440 <= minutes < 4320:
                return token_id
            elif label == "12h" and 720 <= minutes < 1440:
                return token_id
            elif label == "6h" and 360 <= minutes < 720:
                return token_id
            elif label == "3h" and 180 <= minutes < 360:
                return token_id
            elif label == "1h" and 60 <= minutes < 180:
                return token_id
            elif label == "15m" and 15 <= minutes < 60:
                return token_id
            elif label == "5m" and 5 <= minutes < 15:
                return token_id
        
        return None
    
    def process_all_data(self) -> Tuple[Dict[int, List[int]], Dict[str, int]]:
        """Process all OMOP data and return tokenized timelines and vocabulary"""
        logger.info("Processing all OMOP data...")
        logger.info(f"Data directory: {self.omop_data_dir}")
        logger.info(f"Memory limit: {data_config.memory_limit_gb:.1f} GB")
        
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
                        patient_age = 50  # Default age
                else:
                    patient_age = 50  # Default age
                
                tokens = self.tokenize_timeline(timeline, patient_age)
                tokenized_timelines[patient_id] = tokens
            
            # Clear memory after each chunk
            gc.collect()
            
            # Check memory usage
            current_memory = self.get_memory_usage()
            logger.info(f"Memory usage after chunk {i//chunk_size + 1}: {current_memory:.1f} GB")
        
        # Step 7: Save processed data
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
    
    args = parser.parse_args()
    
    # Override config values if provided
    if args.output_dir:
        data_config.output_dir = args.output_dir
    
    if args.memory_limit:
        data_config.memory_limit_gb = args.memory_limit
    
    # Create output directory
    os.makedirs(data_config.output_dir, exist_ok=True)
    
    # Initialize processor with custom data path
    processor = OMOPDataProcessor(data_path=args.data_path)
    
    # Process data if not already processed
    if not os.path.exists(os.path.join(data_config.output_dir, 'tokenized_timelines.pkl')):
        logger.info("Processing new OMOP data...")
        tokenized_timelines, vocab = processor.process_all_data()
    else:
        logger.info("Loading existing processed data...")
        tokenized_timelines, vocab = processor.load_processed_data()
    
    print(f"Processed {len(tokenized_timelines)} patient timelines")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Data saved to: {data_config.output_dir}")

if __name__ == "__main__":
    main()
