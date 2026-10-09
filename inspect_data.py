
import pandas as pd
from pathlib import Path

data_folder = Path("data")

expected_files = [
    "PdM_telemetry.csv",
    "PdM_maint.csv",
    "PdM_machines.csv",
    "PdM_failures.csv",
    "PdM_errors.csv"
]

for filename in expected_files:
    file_path = data_folder / filename

    if not file_path.exists():
        print(f"\nFile not found: {file_path}")
        continue

    df = pd.read_csv(file_path)

    print("\n" + "=" * 60)
    print("FILE:", filename)
    print("Number of rows:", len(df))
    print("Number of columns:", len(df.columns))
    print("Columns:", df.columns.tolist())
    print("\nFirst five rows:")
    print(df.head())
    print("\nMissing values:")
    print(df.isnull().sum())
    print("\nDuplicate rows:", df.duplicated().sum())