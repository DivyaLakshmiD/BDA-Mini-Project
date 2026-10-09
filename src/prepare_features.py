from pathlib import Path
import csv

from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import (
    col, lit, when, sum as spark_sum, last,
    unix_timestamp, to_timestamp, max as spark_max
)

project_dir = Path(__file__).resolve().parent.parent
data_dir = project_dir / "data"
output_dir = data_dir / "processed"
output_file = output_dir / "maintenance_features.csv"

spark = (
    SparkSession.builder
    .appName("PredictiveMaintenanceFeatureEngineering")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

telemetry = spark.read.csv(
    str(data_dir / "PdM_telemetry.csv"),
    header=True,
    inferSchema=True
).withColumn(
    "datetime",
    to_timestamp("datetime", "yyyy-MM-dd HH:mm:ss")
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
).withColumn(
    "datetime",
    to_timestamp("datetime", "yyyy-MM-dd HH:mm:ss")
)

errors = spark.read.csv(
    str(data_dir / "PdM_errors.csv"),
    header=True,
    inferSchema=True
).withColumn(
    "datetime",
    to_timestamp("datetime", "yyyy-MM-dd HH:mm:ss")
)

maintenance = spark.read.csv(
    str(data_dir / "PdM_maint.csv"),
    header=True,
    inferSchema=True
).withColumn(
    "datetime",
    to_timestamp("datetime", "yyyy-MM-dd HH:mm:ss")
)

telemetry = telemetry.join(
    machines,
    on="machineID",
    how="left"
)

telemetry_events = telemetry.select(
    "machineID", "datetime", "volt", "rotate",
    "pressure", "vibration", "model", "age"
).withColumn("event_type", lit("telemetry"))

error_events = errors.select(
    "machineID", "datetime"
).withColumn("event_type", lit("error"))

maintenance_events = maintenance.select(
    "machineID", "datetime"
).withColumn("event_type", lit("maintenance"))

event_columns = telemetry_events.columns

error_events = (
    error_events
    .withColumn("volt", lit(None).cast("double"))
    .withColumn("rotate", lit(None).cast("double"))
    .withColumn("pressure", lit(None).cast("double"))
    .withColumn("vibration", lit(None).cast("double"))
    .withColumn("model", lit(None).cast("string"))
    .withColumn("age", lit(None).cast("integer"))
    .select(event_columns)
)

maintenance_events = (
    maintenance_events
    .withColumn("volt", lit(None).cast("double"))
    .withColumn("rotate", lit(None).cast("double"))
    .withColumn("pressure", lit(None).cast("double"))
    .withColumn("vibration", lit(None).cast("double"))
    .withColumn("model", lit(None).cast("string"))
    .withColumn("age", lit(None).cast("integer"))
    .select(event_columns)
)

timeline = (
    telemetry_events
    .unionByName(error_events)
    .unionByName(maintenance_events)
    .withColumn(
        "event_priority",
        when(col("event_type") == "maintenance", 0)
        .when(col("event_type") == "error", 1)
        .otherwise(2)
    )
)

maintenance_window = (
    Window.partitionBy("machineID")
    .orderBy("datetime", "event_priority")
    .rowsBetween(Window.unboundedPreceding, Window.currentRow)
)

error_window = (
    Window.partitionBy("machineID")
    .orderBy(unix_timestamp("datetime"))
    .rangeBetween(-86400, 0)
)

timeline = timeline.withColumn(
    "last_maintenance_timestamp",
    last(
        when(
            col("event_type") == "maintenance",
            unix_timestamp("datetime")
        ),
        ignorenulls=True
    ).over(maintenance_window)
).withColumn(
    "recent_error_count",
    spark_sum(
        when(col("event_type") == "error", 1).otherwise(0)
    ).over(error_window)
)

features = (
    timeline
    .filter(col("event_type") == "telemetry")
    .withColumn(
        "hours_since_maintenance",
        (
            unix_timestamp("datetime") -
            col("last_maintenance_timestamp")
        ) / 3600
    )
    .fillna({"recent_error_count": 0})
    .drop(
        "event_type",
        "event_priority",
        "last_maintenance_timestamp"
    )
)

features = features.fillna({
    "hours_since_maintenance": 8760,
    "recent_error_count": 0
})

t = features.alias("t")

f = failures.select(
    "machineID",
    col("datetime").alias("failure_datetime")
).alias("f")

failure_matches = t.join(
    f,
    (col("t.machineID") == col("f.machineID")) &
    (
        unix_timestamp(col("f.failure_datetime")) >
        unix_timestamp(col("t.datetime"))
    ) &
    (
        unix_timestamp(col("f.failure_datetime")) <=
        unix_timestamp(col("t.datetime")) + lit(86400)
    ),
    "left"
)

labels = failure_matches.groupBy(
    col("t.machineID").alias("machineID"),
    col("t.datetime").alias("datetime")
).agg(
    spark_max(
        when(col("f.failure_datetime").isNotNull(), 1)
        .otherwise(0)
    ).alias("failure_within_24h")
)

prepared = (
    features
    .join(labels, on=["machineID", "datetime"], how="left")
    .fillna({"failure_within_24h": 0})
    .orderBy("datetime", "machineID")
)

output_dir.mkdir(parents=True, exist_ok=True)

with open(output_file, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(prepared.columns)

    for row in prepared.toLocalIterator():
        writer.writerow(row)

print("\nFEATURE ENGINEERING COMPLETED")
print("Rows:", prepared.count())
print("Columns:", prepared.columns)

print("\nLABEL DISTRIBUTION")
prepared.groupBy("failure_within_24h").count().show()

print("\nMISSING VALUES")
from pyspark.sql.functions import sum as spark_sum
prepared.select([
    spark_sum(
        when(col(c).isNull(), 1).otherwise(0)
    ).alias(c)
    for c in prepared.columns
]).show(truncate=False)

print("\nSAMPLE")
prepared.show(10, truncate=False)

print("\nSaved to:", output_file)

spark.stop()