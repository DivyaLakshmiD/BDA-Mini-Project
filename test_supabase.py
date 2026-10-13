
from pathlib import Path
import tomllib
from supabase import create_client

secrets_path = Path(".streamlit/secrets.toml")

with secrets_path.open("rb") as file:
    secrets = tomllib.load(file)

supabase = create_client(
    secrets["SUPABASE_URL"],
    secrets["SUPABASE_KEY"]
)

try:
    response = (
        supabase.table("maintenance_predictions")
        .select("id")
        .limit(1)
        .execute()
    )
    print("Supabase connection successful!")
    print("Maintenance predictions table is accessible.")
except Exception as error:
    print("Connection or table access failed.")
    print(type(error).__name__, str(error))
