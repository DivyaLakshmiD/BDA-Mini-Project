
from pathlib import Path
import tomllib
import pandas as pd
from supabase import create_client

BASE_DIR = Path(__file__).resolve().parent
SECRETS_PATH = BASE_DIR / ".streamlit" / "secrets.toml"
MACHINES_PATH = BASE_DIR / "data" / "PdM_machines.csv"

with SECRETS_PATH.open("rb") as file:
    secrets = tomllib.load(file)

machines = pd.read_csv(MACHINES_PATH)

machines = machines.rename(
    columns={
        "machineID": "machine_id",
        "model": "machine_model",
        "age": "machine_age",
    }
)

machines["machine_id"] = machines["machine_id"].astype(int)
machines["machine_age"] = machines["machine_age"].astype(int)

records = machines[
    ["machine_id", "machine_model", "machine_age"]
].to_dict(orient="records")

supabase = create_client(
    secrets["SUPABASE_URL"],
    secrets["SUPABASE_KEY"],
)

supabase.auth.sign_in_with_password(
    {
        "email": secrets["SUPABASE_EMAIL"],
        "password": secrets["SUPABASE_PASSWORD"],
    }
)

response = supabase.table("machines").upsert(
    records,
    on_conflict="machine_id",
).execute()

print(f"Machine records processed: {len(records)}")
print("Machine information uploaded successfully.")
