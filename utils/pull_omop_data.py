from pandas_gbq import read_gbq
import pandas as pd
import os
from datetime import date
from google.cloud import bigquery
import numpy as np
from tqdm.notebook import tqdm

dataset_cdr = os.environ['WORKSPACE_CDR']

# Map of table name → date column for filtering
tables_with_dates = {
    "measurement": "measurement_date",
    "condition_occurrence": "condition_start_date",
    "drug_exposure": "drug_exposure_start_date",
    "observation": "observation_date",
    "visit_occurrence": "visit_start_date",
    "visit_detail": "visit_detail_start_date",
    "procedure_occurrence": "procedure_date",
    "device_exposure": "device_exposure_start_date",
    "note": "note_date",
    "death": "death_date",
    "person": None,
    "concept": None,
    "concept_relationship": None,
    "cdm_source": None,
}

def pull_data( 
    start_date = "2021-01-01",
    end_date = "2022-01-01",
    output_root = "omop_data_2021",
    chunk_size = 5_000_000,
    ):

    os.makedirs(output_root, exist_ok=True)

    for table, date_column in tables_with_dates.items():
        print(f"\n🚀 Processing table: {table}")
        
        where_clause = f"WHERE {date_column} BETWEEN DATE('{start_date}') AND DATE('{end_date}')" if date_column else ""
        count_query = f"SELECT COUNT(*) AS total FROM `{dataset_cdr}.{table}` {where_clause}"

        try:
            total_rows = read_gbq(count_query, 
                                dialect="standard", 
                                use_bqstorage_api=True).iloc[0, 0]
            print(f"  ✅ Total rows: {total_rows}")
            
        except Exception as e:
            print(f"  ❌ Failed to count rows for {table}: {e}")
            continue

        if total_rows == 0:
            print(f"  ⚠️ Skipping empty table: {table}")
            continue

        # Step 2: Download in chunks
        offset = 0
        chunk_index = 0
        table_dir = os.path.join(output_root, table)
        os.makedirs(table_dir, exist_ok=True)

        while offset < total_rows:
            print(f"  ⬇️  Downloading rows {offset:,} to {offset + chunk_size:,}")
            query = f"""
            SELECT *
            FROM `{dataset_cdr}.{table}`
            {where_clause}
            LIMIT {chunk_size} OFFSET {offset}
            """

            df_chunk = read_gbq(query, 
                                dialect="standard",
                                use_bqstorage_api=True,
                                progress_bar_type="tqdm")
            chunk_path = os.path.join(table_dir, f"{chunk_index:012}.parquet")
            df_chunk.to_parquet(chunk_path, index=False)
            print(f"  💾 Saved {chunk_path}")

            offset += chunk_size
            chunk_index += 1


pull_data( 
    start_date = "2021-01-01",
    end_date = "2022-01-01",
    output_root = "omop_data_2021",
    )

pull_data( 
    start_date = "2022-01-01",
    end_date = "2023-01-01",
    output_root = "omop_data_2022",
    )

pull_data( 
    start_date = "2023-01-01",
    end_date = "2024-01-01",
    output_root = "omop_data_2023",
    )

pull_data( 
    start_date = "2021-01-01",
    end_date = "2023-01-01",
    output_root = "omop_data_2021_2022",
    )