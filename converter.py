import pandas as pd
import os
from pathlib import Path  # Added for robust path handling

# Set the relative path to your XLSX file here
raw_xlsx_path = r"/Users/jeniphermawia/Downloads/MPESA_Statement_2024-06-03_to_2025-06-03_2547xxxxxx461.xlsx"  # <-- Change this to your XLSX file

# Normalize the path to handle any user input (slashes, backslashes, etc.)
xlsx_path = str(Path(raw_xlsx_path))

# Get the base name without extension
base_name = os.path.splitext(os.path.basename(xlsx_path))[0]

# Set the output CSV file path (same directory as this script)
csv_path = f'{base_name}.csv'

# Define the required columns (headers)
required_columns = [
    'Receipt No.',
    'Completion Time',
    'Details',
    'Transaction Status',
    'Paid In',
    'Withdrawn',
    'Balance'
]

try:
    # Read all sheets into a dict of DataFrames
    all_sheets = pd.read_excel(xlsx_path, sheet_name=None)
    matching_tables = []
    for df in all_sheets.values():
        # Check if all required columns are present (exact match)
        if all(col in df.columns for col in required_columns):
            # Select only the required columns, in the correct order
            matching_tables.append(df[required_columns])
    if matching_tables:
        # Concatenate all matching DataFrames vertically
        combined_df = pd.concat(matching_tables, ignore_index=True)
        # Write to CSV with the required columns as header
        combined_df.to_csv(csv_path, index=False)
        print(f'Converted matching tables in {xlsx_path} to {csv_path}')
    else:
        print('No tables with the required columns were found.')
except Exception as e:
    print(f'Error: {e}')
