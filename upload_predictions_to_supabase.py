from pathlib import Path
import ast
import tomllib
import pandas as pd
from supabase import create_client

BASE_DIR = Path(__file__).resolve().parent

with (BASE_DIR / ".streamlit" / "secrets.toml").open("rb") as file:
    secrets = tomllib.load(file)

supabase = create_client(
    secrets["SUPABASE_URL"],
    secrets["SUPABASE_KEY"]
)

auth = supabase.auth.sign_in_with_password({
    "email": secrets["SUPABASE_EMAIL"],
    "password": secrets["SUPABASE_PASSWORD"]
})


if not auth.user or not auth.session:
    raise RuntimeError("Supabase authentication failed.")

supabase.postgrest.auth(auth.session.access_token)

print("Authenticated with Supabase.")
print("Authenticated user ID:", auth.user.id)
print("User access token attached to database client.")

predictions = pd.read_csv(BASE_DIR / "results" / "test_predictions.csv")
telemetry = pd.read_csv(BASE_DIR / "data" / "PdM_telemetry.csv")
machines = pd.read_csv(BASE_DIR / "data" / "PdM_machines.csv")

predictions["datetime"] = pd.to_datetime(predictions["datetime"])
telemetry["datetime"] = pd.to_datetime(telemetry["datetime"])

predictions["machineID"] = predictions["machineID"].astype(int)
telemetry["machineID"] = telemetry["machineID"].astype(int)
machines["machineID"] = machines["machineID"].astype(int)

sample = (
    predictions.sort_values(["datetime", "machineID"])
    .head(5)
    .copy()
)

machine_lookup = machines.set_index("machineID").to_dict("index")
telemetry_by_machine = {
    int(machine_id): group.sort_values("datetime")
    for machine_id, group in telemetry.groupby("machineID")
}

records = []

for _, row in sample.iterrows():
    machine_id = int(row["machineID"])
    recorded_at = row["datetime"]

    probability_values = ast.literal_eval(str(row["probability"]))
    failure_probability = float(probability_values[1])

    if failure_probability >= 0.70:
        risk_level = "HIGH"
    elif failure_probability >= 0.40:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    machine = machine_lookup.get(machine_id, {})
    machine_telemetry = telemetry_by_machine.get(machine_id)

    sensor = None
    if machine_telemetry is not None:
        prior = machine_telemetry[
            machine_telemetry["datetime"] <= recorded_at
        ]
        if not prior.empty:
            sensor = prior.iloc[-1]

    record = {
        "machine_id": machine_id,
        "recorded_at": recorded_at.strftime("%Y-%m-%d %H:%M:%S"),
        "failure_probability": failure_probability,
        "risk_level": risk_level,
        "predicted_failure": bool(float(row["prediction"]) >= 0.5),
        "machine_model": machine.get("model"),
        "machine_age": int(machine["age"]) if pd.notna(machine.get("age")) else None,
        "recent_error_count": None,
        "hours_since_maintenance": None,
        "voltage": float(sensor["volt"]) if sensor is not None else None,
        "rotation": float(sensor["rotate"]) if sensor is not None else None,
        "pressure": float(sensor["pressure"]) if sensor is not None else None,
        "vibration": float(sensor["vibration"]) if sensor is not None else None,
    }

    records.append(record)

print("Prepared", len(records), "records.")
print("Machines:", [record["machine_id"] for record in records])

response = (
    supabase.table("maintenance_predictions")
    .insert(records, returning="minimal")
    .execute()
)

print("Upload request completed.")
print("Check Supabase Table Editor for the five inserted records.")

print("Upload request completed.")
print("Check Supabase Table Editor for the five inserted records.")
