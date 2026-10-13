
from pathlib import Path

import pandas as pd
from supabase import create_client
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "data" / "PdM_telemetry.csv"


client = create_client(
    st.secrets["SUPABASE_URL"],
    st.secrets["SUPABASE_KEY"]
)


BATCH_SIZE = 500
total_uploaded = 0

for chunk in pd.read_csv(CSV_PATH, chunksize=BATCH_SIZE):
    chunk = chunk.rename(columns={
        "machineID": "machine_id",
        "datetime": "recorded_at",
        "volt": "voltage",
        "rotate": "rotation",
    })

    chunk["recorded_at"] = pd.to_datetime(
        chunk["recorded_at"]
    ).dt.strftime("%Y-%m-%dT%H:%M:%S")

    records = chunk[
        [
            "machine_id",
            "recorded_at",
            "voltage",
            "rotation",
            "pressure",
            "vibration",
        ]
    ].where(pd.notna(chunk), None).to_dict("records")

    client.table("machine_telemetry").upsert(
        records,
        on_conflict="machine_id,recorded_at"
    ).execute()

    total_uploaded += len(records)
    print(f"Processed {total_uploaded:,} telemetry records")

print(f"Finished. Total records processed: {total_uploaded:,}")
