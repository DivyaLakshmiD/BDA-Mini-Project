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
    response = supabase.auth.sign_in_with_password({
        "email": secrets["SUPABASE_EMAIL"],
        "password": secrets["SUPABASE_PASSWORD"]
    })

    if response.user:
        print("Supabase user authentication successful!")
        print("User ID:", response.user.id)
    else:
        print("Authentication returned no user.")

except Exception as error:
    print("Authentication failed.")
    print(type(error).__name__, str(error))
