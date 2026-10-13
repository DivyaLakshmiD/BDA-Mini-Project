
from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, from_json
from pyspark.sql.types import (
    StructType,
    StructField,
    StringType,
    IntegerType,
    DoubleType
)

PROJECT_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_DIR / "outputs" / "live_sensor_stream"
CHECKPOINT_DIR = PROJECT_DIR / "outputs" / "kafka_json_checkpoint"

KAFKA_SERVER = "localhost:9092"
TOPIC_NAME = "machine-sensor-stream"

schema = StructType([
    StructField("datetime", StringType(), True),
    StructField("machineID", IntegerType(), True),
    StructField("volt", DoubleType(), True),
    StructField("rotate", DoubleType(), True),
    StructField("pressure", DoubleType(), True),
    StructField("vibration", DoubleType(), True)
])

spark = None
query = None

try:
    print("Starting Spark Structured Streaming...")

    spark = (
        SparkSession.builder
        .appName("BDA_Predictive_Maintenance_Streaming")
        .config(
            "spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.13:4.2.0"
        )
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    kafka_stream = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", KAFKA_SERVER)
        .option("subscribe", TOPIC_NAME)
        .option("startingOffsets", "latest")
        .load()
    )

    parsed_stream = (
        kafka_stream
        .select(
            from_json(
                col("value").cast("string"),
                schema
            ).alias("sensor")
        )
        .select("sensor.*")
        .filter(col("machineID").isNotNull())
    )

    query = (
        parsed_stream.writeStream
        .format("json")
        .outputMode("append")
        .option("path", str(OUTPUT_DIR))
        .option("checkpointLocation", str(CHECKPOINT_DIR))
        .start()
    )

    print("Kafka connection established.")
    print(f"Reading topic: {TOPIC_NAME}")
    print(f"Saving sensor records to: {OUTPUT_DIR}")
    print("Waiting for incoming sensor records...")
    print("Press Ctrl+C to stop.")

    query.awaitTermination()

except KeyboardInterrupt:
    print("Stopping the streaming query...")

finally:
    if query is not None:
        try:
            query.stop()
        except Exception:
            pass

    if spark is not None:
        try:
            spark.stop()
        except Exception:
            pass

    print("Streaming application stopped.")