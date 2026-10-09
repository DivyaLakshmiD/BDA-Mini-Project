
from pathlib import Path
import csv

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, min as spark_min, when, unix_timestamp, to_timestamp

project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir / "data"
output_dir = data_dir / "processed"
output_file = output_dir / "telemetry_labeled.csv"

spark = (
    SparkSession.builder
    .appName("PredictiveMaintenancePreparation")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("WARN")

telemetry = spark.read.csv(
    str(data_dir / "PdM_telemetry.csv"),
    header=True,
    inferSchema=True
)

machines = spark.read.csv(
    str(data_dir / "PdM_machines.csv"),
    header=True,
    inferSchema=True
)

failures = spark.read.csv(
    str(data_dir / "PdM_failures.csv"),
    header=True,
    inferSchema=True
)

telemetry = telemetry.withColumn(
    "datetime",
    to_timestamp("datetime", "yyyy-MM-dd HH:mm:ss")
)

failures = failures.withColumn(
    "datetime",
    to_timestamp("datetime", "yyyy-MM-dd HH:mm:ss")
)

joined = telemetry.join(
    machines,
    on="machineID",
    how="left"
)

failure_events = failures.select(
    "machineID",
    col("datetime").alias("failure_datetime")
)

future_failures = joined.join(
    failure_events,
    on="machineID",
    how="left"
).filter(
    (col("failure_datetime") > col("datetime")) &
    (
        unix_timestamp("failure_datetime") -
        unix_timestamp("datetime") <= 86400
    )
)

labels = future_failures.groupBy(
    "machineID",
    "datetime"
).agg(
    spark_min("failure_datetime").alias("next_failure_datetime")
)

prepared = joined.join(
    labels,
    on=["machineID", "datetime"],
    how="left"
).withColumn(
    "failure_within_24h",
    when(col("next_failure_datetime").isNotNull(), 1).otherwise(0)
).drop("next_failure_datetime")

output_dir.mkdir(parents=True, exist_ok=True)

with open(output_file, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(prepared.columns)

    for row in prepared.toLocalIterator():
        writer.writerow(row)

print("\nDATA PREPARATION COMPLETED")
print("Prepared rows:", prepared.count())
print("Columns:", prepared.columns)

prepared.groupBy("failure_within_24h").count().show()

prepared.select(
    "datetime",
    "machineID",
    "volt",
    "rotate",
    "pressure",
    "vibration",
    "model",
    "age",
    "failure_within_24h"
).show(10, truncate=False)

print("Saved file:", output_file)

spark.stop()