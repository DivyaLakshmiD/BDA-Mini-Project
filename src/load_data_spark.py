
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import to_timestamp, col

project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir / "data"

spark = (
    SparkSession.builder
    .appName("PredictiveMaintenance")
    .master("local[*]")
    .getOrCreate()
)

file_names = {
    "telemetry": "PdM_telemetry.csv",
    "maintenance": "PdM_maint.csv",
    "machines": "PdM_machines.csv",
    "failures": "PdM_failures.csv",
    "errors": "PdM_errors.csv"
}

datasets = {}

for name, filename in file_names.items():
    path = data_dir / filename

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = spark.read.csv(str(path), header=True, inferSchema=True)

    if "datetime" in df.columns:
        df = df.withColumn(
            "datetime",
            to_timestamp(col("datetime"), "yyyy-MM-dd HH:mm:ss")
        )

    datasets[name] = df

    print(f"\n{name.upper()}")
    print("Rows:", df.count())
    print("Columns:", df.columns)
    df.show(5, truncate=False)
    df.printSchema()

spark.stop()