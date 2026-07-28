import os
import pandas as pd
from tqdm import tqdm

# Configuration
large_file = "data\\XLON_membership\\compustat_returns.csv"
output_file = "data\\XLON_membership\\compustat_filtered.csv"

# Filters we use
xlon_exchange_code = "194"   # XLON
ordinary_share_code = "0"    # ordinary share

# Get total file size in bytes
file_size_bytes = os.path.getsize(large_file)

print("Step 1: Filtering the file for XLON ordinary shares...")

first_chunk = True

# Track progress by actual file bytes consumed rather than dataframe memory use.
with open(large_file, "rb") as infile, tqdm(
    total=file_size_bytes,
    unit="B",
    unit_scale=True,
    desc="Filtering Rows",
) as pbar:
    last_position = 0
    for chunk in pd.read_csv(infile, chunksize=100000, low_memory=False, dtype=str):
        # Standardize column names just in case
        chunk.columns = chunk.columns.str.lower().str.strip()

        required_columns = {"exchg", "tpci"}
        missing = required_columns.difference(chunk.columns)
        if missing:
            raise KeyError(f"Missing required columns: {sorted(missing)}")

        # Standardize filter columns
        chunk["exchg"] = chunk["exchg"].astype("string").str.strip()
        chunk["tpci"] = chunk["tpci"].astype("string").str.strip()

        # Apply filters
        filtered_chunk = chunk[
            (chunk["exchg"] == xlon_exchange_code) &
            (chunk["tpci"] == ordinary_share_code)
        ]

        # Write output incrementally
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
