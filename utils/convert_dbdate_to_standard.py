#!/usr/bin/env python3
"""
Convert All of Us OMOP parquet files with dbdate types to standard parquet format
This script handles the dbdate conversion issue by reading and rewriting the data
"""

import os
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import gc

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_dbdate_column(df, col_name):
    """Convert a dbdate column to standard datetime"""
    if col_name not in df.columns:
        return df
    
    logger.info(f"Converting dbdate column: {col_name}")
    
    # Sample values to understand the format
    sample_values = df[col_name].dropna().head(5).tolist()
    logger.info(f"Sample values from {col_name}: {sample_values}")
    
    # Try different conversion strategies
    conversion_success = False
    
    # Strategy 1: Try as integer days since epoch (most common for All of Us)
    try:
        temp_df = df.copy()
        temp_df[col_name] = pd.to_datetime(temp_df[col_name], unit='D', errors='coerce')
        if temp_df[col_name].notna().sum() > 0:
            df[col_name] = temp_df[col_name]
            logger.info(f"Successfully converted {col_name} from days since epoch")
            conversion_success = True
    except Exception as e:
        logger.info(f"Days since epoch conversion failed for {col_name}: {e}")
    
    # Strategy 2: Try as integer seconds since epoch
    if not conversion_success:
        try:
            temp_df = df.copy()
            temp_df[col_name] = pd.to_datetime(temp_df[col_name], unit='s', errors='coerce')
            if temp_df[col_name].notna().sum() > 0:
                df[col_name] = temp_df[col_name]
                logger.info(f"Successfully converted {col_name} from seconds since epoch")
                conversion_success = True
        except Exception as e:
            logger.info(f"Seconds since epoch conversion failed for {col_name}: {e}")
    
    # Strategy 3: Try as integer milliseconds since epoch
    if not conversion_success:
        try:
            temp_df = df.copy()
            temp_df[col_name] = pd.to_datetime(temp_df[col_name], unit='ms', errors='coerce')
            if temp_df[col_name].notna().sum() > 0:
                df[col_name] = temp_df[col_name]
                logger.info(f"Successfully converted {col_name} from milliseconds since epoch")
                conversion_success = True
        except Exception as e:
            logger.info(f"Milliseconds since epoch conversion failed for {col_name}: {e}")
    
    # Strategy 4: Try string parsing
    if not conversion_success:
        try:
            temp_df = df.copy()
            temp_df[col_name] = pd.to_datetime(temp_df[col_name], errors='coerce')
            if temp_df[col_name].notna().sum() > 0:
                df[col_name] = temp_df[col_name]
                logger.info(f"Successfully converted {col_name} from string parsing")
                conversion_success = True
        except Exception as e:
            logger.info(f"String parsing conversion failed for {col_name}: {e}")
    
    if not conversion_success:
        logger.warning(f"Could not convert {col_name} to datetime, keeping as object")
    
    return df

def convert_table(input_file, output_file, table_name):
    """Convert a single table file from dbdate to standard format"""
    try:
        logger.info(f"Converting {table_name} table...")
        
        # Read the parquet file with PyArrow
        table = pq.read_table(input_file)
        logger.info(f"Successfully loaded {table_name} with {len(table)} rows")
        
        # Get the original schema
        original_schema = table.schema
        logger.info(f"Original schema: {original_schema}")
        
        # Handle schema modifications
        schema_changed = False
        new_fields = []
        
        for field in original_schema:
            field_type = field.type
            field_name = field.name
            
            # Handle date types (including dbdate)
            if (pa.types.is_date32(field_type) or 
                pa.types.is_date64(field_type) or
                str(field_type) == 'dbdate'):
                
                logger.info(f"Converting date field '{field_name}' from {field_type} to timestamp")
                new_fields.append(pa.field(field_name, pa.timestamp('ns')))
                schema_changed = True
                
            # Handle other problematic types
            elif pa.types.is_time32(field_type) or pa.types.is_time64(field_type):
                logger.info(f"Converting time field '{field_name}' from {field_type} to timestamp")
                new_fields.append(pa.field(field_name, pa.timestamp('ns')))
                schema_changed = True
                
            # Keep other fields as-is
            else:
                new_fields.append(field)
        
        # Apply schema changes if needed
        if schema_changed:
            logger.info("Applying schema modifications...")
            new_schema = pa.schema(new_fields)
            table = table.cast(new_schema)
            logger.info(f"New schema: {new_schema}")
        else:
            logger.info("No schema changes needed")
        
        # Convert to pandas for additional processing
        df = table.to_pandas()
        
        # Handle any remaining data type issues in pandas
        df = fix_remaining_data_types(df, table_name)
        
        # Save the converted data
        logger.info(f"Saving converted {table_name} to {output_file}")
        df.to_parquet(output_file, engine='pyarrow', index=False)
        
        logger.info(f"Successfully converted {table_name}: {len(df)} rows")
        return True
        
    except Exception as e:
        logger.error(f"Error converting {table_name}: {e}")
        return False

def fix_remaining_data_types(df, table_name):
    """Fix any remaining data type issues after PyArrow conversion"""
    logger.info(f"Fixing remaining data types for {table_name}")
    
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

def convert_all_tables(input_dir, output_dir, tables=None):
    """Convert all OMOP tables from dbdate to standard format"""
    
    # Default tables to convert
    if tables is None:
        tables = [
            'person', 'visit_occurrence', 'condition_occurrence', 'drug_exposure',
            'procedure_occurrence', 'measurement', 'observation', 'death'
        ]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    successful_conversions = 0
    total_tables = len(tables)
    
    logger.info(f"Starting conversion of {total_tables} tables...")
    logger.info(f"Input directory: {input_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    for table_name in tables:
        input_file = os.path.join(input_dir, table_name, '000000000000.parquet')
        output_file = os.path.join(output_dir, f'{table_name}.parquet')
        
        if not os.path.exists(input_file):
            logger.warning(f"Input file not found: {input_file}")
            continue
        
        # Create output subdirectory
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        if convert_table(input_file, output_file, table_name):
            successful_conversions += 1
        
        # Clear memory
        gc.collect()
    
    logger.info(f"Conversion completed: {successful_conversions}/{total_tables} tables successful")
    return successful_conversions == total_tables

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Convert All of Us OMOP parquet files with dbdate to standard format')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory containing OMOP data with dbdate types')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Output directory for converted standard parquet files')
    parser.add_argument('--tables', type=str, nargs='+', default=None,
                       help='Specific tables to convert (default: all standard OMOP tables)')
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory does not exist: {args.input_dir}")
        return
    
    # Convert tables
    success = convert_all_tables(args.input_dir, args.output_dir, args.tables)
    
    if success:
        logger.info("✅ All tables converted successfully!")
        logger.info(f"📁 Converted data saved to: {args.output_dir}")
        logger.info(f"💡 You can now use this directory with the data processor:")
        logger.info(f"   python data_processor.py --data_path {args.output_dir} --tag your_tag")
    else:
        logger.warning("⚠️  Some tables failed to convert. Check the logs above.")
        logger.info(f"📁 Partial data saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
