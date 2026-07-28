import os
from pathlib import Path
import sys

import pandas as pd
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.pipeline_config import FORMATION_YEARS

# Configuration
large_file = "data\\XLON_membership\\compustat_filtered.csv"
output_file = "data\\XLON_membership\\compustat_filtered_june.csv"
date_column = "datadate"
file_size_bytes = os.path.getsize(large_file)

print("Step 1: Extracting unique available dates...")

unique_dates = set()

with open(large_file, "rb") as infile, tqdm(
    total=file_size_bytes,
    unit="B",
    unit_scale=True,
    desc="Reading Dates",
) as pbar:
    last_position = 0
    for chunk in pd.read_csv(infile, usecols=[date_column], chunksize=500000, dtype=str):
        dates = pd.to_datetime(chunk[date_column], errors="coerce").dropna().dt.normalize()
        unique_dates.update(dates.tolist())

        current_position = infile.tell()
        pbar.update(current_position - last_position)
        last_position = current_position

    if last_position < file_size_bytes:
        pbar.update(file_size_bytes - last_position)

available_dates = pd.Series(sorted(unique_dates))

configured_years = set(FORMATION_YEARS)
available_dates = available_dates[available_dates.dt.year.isin(configured_years)].reset_index(drop=True)

print(f"\nFound {len(available_dates)} unique trading dates in configured formation years.")
print("Step 2: Collecting all available dates from June 20 to June 30 for each configured year...")

target_dates = set()

for year in sorted(available_dates.dt.year.unique()):
    start_date = pd.Timestamp(f"{year}-06-20")
    end_date = pd.Timestamp(f"{year}-06-30")

    dates_in_window = available_dates[
        (available_dates.dt.year == year)
        & (available_dates >= start_date)
        & (available_dates <= end_date)
    ]

    target_dates.update(dates_in_window.tolist())

print("Chosen dates:")
for d in sorted(target_dates):
    print(d.strftime("%Y-%m-%d"))

print("\nStep 3: Filtering rows for configured years and June 20 to June 30 dates...")

first_chunk = True

with open(large_file, "rb") as infile, tqdm(
    total=file_size_bytes,
    unit="B",
    unit_scale=True,
    desc="Filtering Rows",
) as pbar:
    last_position = 0
    for chunk in pd.read_csv(infile, chunksize=100000, low_memory=False, dtype=str):
        chunk.columns = chunk.columns.str.lower().str.strip()

        if date_column not in chunk.columns:
            raise KeyError(f"Missing required date column: {date_column}")

        parsed_dates = pd.to_datetime(chunk[date_column], errors="coerce").dt.normalize()
        filtered_chunk = chunk[parsed_dates.isin(target_dates)]

        if not filtered_chunk.empty:
            if first_chunk:
                filtered_chunk.to_csv(output_file, index=False, mode="w")
                first_chunk = False
            else:
                filtered_chunk.to_csv(output_file, index=False, mode="a", header=False)

        current_position = infile.tell()
        pbar.update(current_position - last_position)
        last_position = current_position

    if last_position < file_size_bytes:
        pbar.update(file_size_bytes - last_position)

print(f"\nSuccess! Filtered data saved to {output_file}")
