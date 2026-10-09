
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as spark_sum, min, max

project_dir = Path(__file__).resolve().parent.parent
data_file = project_dir / "data" / "telemetry_labeled.csv"

spark = (
    SparkSession.builder
    .appName("ValidatePredictiveMaintenanceData")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

df = spark.read.csv(
    str(data_file),
    header=True,
    inferSchema=True
)

print("\nDATASET SIZE")
print("Rows:", df.count())
print("Columns:", len(df.columns))

print("\nMISSING VALUES PER COLUMN")
df.select([
    spark_sum(col(c).isNull().cast("int")).alias(c)
    for c in df.columns
]).show(truncate=False)

print("\nFAILURE LABEL DISTRIBUTION")
df.groupBy("failure_within_24h").count().show()

print("\nFAILURE LABEL PERCENTAGES")
df.groupBy("failure_within_24h").count().withColumn(
    "percentage",
    col("count") * 100 / df.count()
).show()

print("\nDATE RANGE")
df.select(
    min("datetime").alias("start_datetime"),
    max("datetime").alias("end_datetime")
).show(truncate=False)

print("\nDUPLICATE RECORD COUNT")
print(
    "Duplicates:",
    df.count() - df.dropDuplicates(["machineID", "datetime"]).count()
)

spark.stop()