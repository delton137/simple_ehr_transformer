#!/usr/bin/env python3
"""
Fix dbdate issues in All of Us OMOP parquet files
This script handles the custom dbdate type and other date type issues
"""

import pyarrow as pa
import pyarrow.parquet as pq
from glob import glob
import os
import argparse
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_dbdate_files(input_root, output_root=None, overwrite=False):
    """
    Fix dbdate issues in parquet files
    
    Args:
        input_root: Directory containing OMOP data with subdirectories
        output_root: Output directory (if None, will overwrite in place)
        overwrite: Whether to overwrite original files
    """
    
    if output_root is None and not overwrite:
        raise ValueError("Must specify output_root or set overwrite=True")
    
    if output_root and not os.path.exists(output_root):
        os.makedirs(output_root, exist_ok=True)
        logger.info(f"Created output directory: {output_root}")
    
    # Find all parquet files
    parquet_files = sorted(glob(f"{input_root}/*/*.parquet"))
    logger.info(f"Found {len(parquet_files)} parquet files to process")
    
    if not parquet_files:
        logger.warning(f"No parquet files found in {input_root}/*/*.parquet")
        return
    
    processed_files = 0
    errors = 0
    
    for file_path in parquet_files:
        try:
            logger.info(f"Processing: {file_path}")
            
            # Read the parquet file
            table = pq.read_table(file_path)
            logger.info(f"  Loaded table with {len(table)} rows and {len(table.column_names)} columns")
            
            # Get the original schema
            original_schema = table.schema
            logger.info(f"  Original schema: {original_schema}")
            
            # Handle column renaming
            if "discharge_to_concept_id" in table.column_names:
                logger.info("  Renaming 'discharge_to_concept_id' to 'discharged_to_concept_id'")
                new_cols = [
                    "discharged_to_concept_id" if name == "discharge_to_concept_id" else name
                    for name in table.column_names
                ]
                table = table.rename_columns(new_cols)
            
            # Handle problematic date types including dbdate
            schema = table.schema
            new_fields = []
            schema_changed = False
            
            for field in schema:
                field_type = field.type
                field_name = field.name
                
                # Check for dbdate type (string comparison)
                if str(field_type) == 'dbdate':
                    logger.info(f"  Converting dbdate field '{field_name}' to timestamp")
                    new_fields.append(pa.field(field_name, pa.timestamp('ns')))
                    schema_changed = True
                
                # Handle other date types
                elif pa.types.is_date32(field_type) or pa.types.is_date64(field_type):
                    logger.info(f"  Converting date field '{field_name}' from {field_type} to timestamp")
                    new_fields.append(pa.field(field_name, pa.timestamp('ns')))
                    schema_changed = True
                
                # Handle time types
                elif pa.types.is_time32(field_type) or pa.types.is_time64(field_type):
                    logger.info(f"  Converting time field '{field_name}' from {field_type} to timestamp")
                    new_fields.append(pa.field(field_name, pa.timestamp('ns')))
                    schema_changed = True
                
                # Keep other fields as-is
                else:
                    new_fields.append(field)
            
            # Apply schema changes if needed
            if schema_changed:
                logger.info("  Applying schema modifications...")
                new_schema = pa.schema(new_fields)
                table = table.cast(new_schema)
                logger.info(f"  New schema: {new_schema}")
            else:
                logger.info("  No schema changes needed")
            
            # Determine output path
            if output_root:
                # Create subdirectory structure in output
                rel_path = os.path.relpath(file_path, input_root)
                output_path = os.path.join(output_root, rel_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            else:
                output_path = file_path
            
            # Write the fixed table
            logger.info(f"  Writing to: {output_path}")
            pq.write_table(table, output_path)
            
            processed_files += 1
            logger.info(f"  ✅ Successfully processed {file_path}")
            
        except Exception as e:
            errors += 1
            logger.error(f"  ❌ Error processing {file_path}: {e}")
            continue
    
    logger.info(f"\nProcessing complete:")
    logger.info(f"  ✅ Successfully processed: {processed_files} files")
    if errors > 0:
        logger.warning(f"  ❌ Errors: {errors} files")
    
    if output_root:
        logger.info(f"  📁 Fixed files saved to: {output_root}")
    else:
        logger.info(f"  🔄 Original files updated in place")

def main():
    parser = argparse.ArgumentParser(description='Fix dbdate issues in All of Us OMOP parquet files')
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory containing OMOP data with subdirectories')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for fixed files (if not specified, will overwrite in place)')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite original files (use with caution)')
    
    args = parser.parse_args()
    
    # Validate input directory
    if not os.path.exists(args.input_dir):
        logger.error(f"Input directory does not exist: {args.input_dir}")
        return
    
    # Check if we have subdirectories with parquet files
    subdirs = [d for d in os.listdir(args.input_dir) 
               if os.path.isdir(os.path.join(args.input_dir, d))]
    
    if not subdirs:
        logger.error(f"No subdirectories found in {args.input_dir}")
        logger.info("Expected structure: input_dir/table_name/*.parquet")
        return
    
    logger.info(f"Found subdirectories: {subdirs}")
    
    # Process the files
    try:
        fix_dbdate_files(args.input_dir, args.output_dir, args.overwrite)
        logger.info("✅ Dbdate fixing completed successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to fix dbdate files: {e}")

if __name__ == "__main__":
    main()
