#!/usr/bin/env python3
"""
Count unique persons (person_id) with any data in OMOP folders.

Strategy:
- For each dataset directory, scan common OMOP tables that include person_id
  and union all person_id values to count unique persons with any data.
- Also report unique persons present in the person table if available.

Uses Polars for efficient lazy scanning if installed; falls back to PyArrow/pandas.
"""

import os
import sys
from typing import List, Set, Optional


def _tables_to_scan() -> List[str]:
    # Common OMOP tables that reference person_id (subset for speed; extend as needed)
    return [
        "person",
        "visit_occurrence",
        "visit_detail",
        "condition_occurrence",
        "drug_exposure",
        "procedure_occurrence",
        "measurement",
        "observation",
        "device_exposure",
        "death",
    ]


def _has_any_parquet(dir_path: str) -> bool:
    try:
        for fn in os.listdir(dir_path):
            if fn.endswith(".parquet"):
                return True
    except FileNotFoundError:
        return False
    return False


def _collect_person_ids_polars(dataset_dir: str, tables: List[str]) -> (Set[int], Set[int]):
    import polars as pl

    lfs_any = []
    person_ids_from_person = set()

    for table in tables:
        table_dir = os.path.join(dataset_dir, table)
        if not os.path.isdir(table_dir) or not _has_any_parquet(table_dir):
            continue
        pattern = os.path.join(table_dir, "*.parquet")
        try:
            scan = pl.scan_parquet(pattern)
            names = scan.collect_schema().names()
        except Exception:
            continue
        if "person_id" not in names:
            continue
        lf = scan.select([pl.col("person_id").cast(pl.Int64).alias("person_id")])
        if table == "person":
            # For person table, we track separately
            try:
                person_ids_from_person = set(
                    lf.unique().collect().get_column("person_id").to_list()
                )
            except Exception:
                person_ids_from_person = set()
        else:
            lfs_any.append(lf)

    # Union across all non-person tables to get persons with any clinical data
    persons_with_any: Set[int] = set()
    if lfs_any:
        try:
            combined = pl.concat(lfs_any, how="vertical_relaxed").select(["person_id"]).unique()
            persons_with_any = set(combined.collect().get_column("person_id").to_list())
        except Exception:
            persons_with_any = set()

    return persons_with_any, person_ids_from_person


def _collect_person_ids_arrow(dataset_dir: str, tables: List[str]) -> (Set[int], Set[int]):
    # Fallback using pyarrow.dataset / pandas
    persons_with_any: Set[int] = set()
    person_ids_from_person: Set[int] = set()
    try:
        import pyarrow.dataset as ds
        use_arrow = True
    except Exception:
        use_arrow = False

    for table in tables:
        table_dir = os.path.join(dataset_dir, table)
        if not os.path.isdir(table_dir) or not _has_any_parquet(table_dir):
            continue
        if use_arrow:
            try:
                dataset = ds.dataset(table_dir, format="parquet")
                schema = dataset.schema
                if "person_id" not in [f.name for f in schema]:
                    continue
                t = dataset.to_table(columns=["person_id"])  # only read one column
                ids = set(t.column(0).to_pylist())
            except Exception:
                ids = set()
        else:
            # Last resort: iterate files with pandas
            try:
                import pandas as pd
                ids = set()
                for fn in os.listdir(table_dir):
                    if not fn.endswith(".parquet"):
                        continue
                    fp = os.path.join(table_dir, fn)
                    try:
                        df = pd.read_parquet(fp, columns=["person_id"])  # type: ignore[arg-type]
                        ids.update(df["person_id"].dropna().astype("int64").unique().tolist())
                    except Exception:
                        continue
            except Exception:
                ids = set()

        if table == "person":
            person_ids_from_person = ids
        else:
            persons_with_any.update(ids)

    return persons_with_any, person_ids_from_person


def count_unique_persons(dataset_dir: str) -> (int, Optional[int]):
    tables = _tables_to_scan()
    # Prefer Polars if available
    try:
        import polars as _pl  # noqa: F401
        persons_with_any, persons_from_person = _collect_person_ids_polars(dataset_dir, tables)
    except Exception:
        persons_with_any, persons_from_person = _collect_person_ids_arrow(dataset_dir, tables)
    any_count = len({pid for pid in persons_with_any if pid is not None})
    person_table_count = len({pid for pid in persons_from_person if pid is not None}) if persons_from_person else None
    return any_count, person_table_count


def main(argv: List[str]) -> int:
    import argparse

    default_dirs = [
        "./utils/omop_data_2022",
        "./utils/omop_data_2021_2022",
        "./utils/omop_data_2023",
        "./utils/omop_data_2024",
        "./utils/omop_data_2021_2022_2023",
    ]

    parser = argparse.ArgumentParser(description="Count unique persons with data in OMOP folders")
    parser.add_argument("data_dirs", nargs="*", default=default_dirs, help="OMOP dataset directories")
    args = parser.parse_args(argv)

    for d in args.data_dirs:
        if not os.path.isdir(d):
            print(f"{d}: directory not found")
            continue
        any_count, person_count = count_unique_persons(d)
        if person_count is not None:
            print(f"{d}: persons_with_any_data={any_count:,} | persons_in_person_table={person_count:,}")
        else:
            print(f"{d}: persons_with_any_data={any_count:,}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


