from pathlib import Path

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col,
    when
)
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array

PROJECT_DIR = Path(__file__).resolve().parent.parent

STREAM_DIR = PROJECT_DIR / "outputs" / "live_sensor_stream"
FEATURES_FILE = PROJECT_DIR / "data" / "processed" / "maintenance_features.csv"
MODEL_DIR = PROJECT_DIR / "models" / "predictive_maintenance_rf"
OUTPUT_DIR = PROJECT_DIR / "outputs" / "live_predictions"

spark = None

try:
    if not STREAM_DIR.exists():
        raise FileNotFoundError(
            f"Sensor stream folder not found: {STREAM_DIR}"
        )

    if not FEATURES_FILE.exists():
        raise FileNotFoundError(
            f"Processed features file not found: {FEATURES_FILE}"
        )

    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_DIR}"
        )

    spark = (
        SparkSession.builder
        .appName("BDA_Live_Maintenance_Predictions")
        .master("local[*]")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")

    print("Reading saved Kafka sensor records...")

    
    json_files = [
        str(file)
        for file in STREAM_DIR.glob("*.json")
        if file.stat().st_size > 0
    ]

    if not json_files:
        raise RuntimeError(
            f"No non-empty JSON sensor files found in {STREAM_DIR}"
        )

    sensor_data = (
        spark.read
        .json(json_files)
        .withColumn("datetime", col("datetime").cast("timestamp"))
    )


    print("Reading existing maintenance features...")

    features = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(str(FEATURES_FILE))
        .withColumn("datetime", col("datetime").cast("timestamp"))
    )

    sensor_data = sensor_data.alias("s")
    features = features.alias("f")

    joined = sensor_data.join(
        features,
        (
            (col("s.machineID") == col("f.machineID")) &
            (col("s.datetime") == col("f.datetime"))
        ),
        "inner"
    )

    prediction_input = joined.select(
        col("f.machineID").alias("machineID"),
        col("f.datetime").alias("datetime"),
        col("s.volt").alias("volt"),
        col("s.rotate").alias("rotate"),
        col("s.pressure").alias("pressure"),
        col("s.vibration").alias("vibration"),
        col("f.model").alias("model"),
        col("f.age").alias("age"),
        col("f.recent_error_count").alias("recent_error_count"),
        col("f.hours_since_maintenance").alias(
            "hours_since_maintenance"
        )
    ).dropDuplicates(["machineID", "datetime"])

    matched_count = prediction_input.count()

    print("Matched sensor records:", matched_count)

    if matched_count == 0:
        raise RuntimeError(
            "No sensor records matched the processed dataset. "
            "Check the datetime and machineID values."
        )

    print("Loading saved Random Forest pipeline...")

    model = PipelineModel.load(str(MODEL_DIR))

    print("Generating maintenance predictions...")

    predictions = (
        model.transform(prediction_input)
        .withColumn(
            "failure_probability",
            vector_to_array(col("probability"))[1]
        )
        .withColumn(
            "risk_level",
            when(col("failure_probability") >= 0.70, "HIGH")
            .when(col("failure_probability") >= 0.40, "MEDIUM")
            .otherwise("LOW")
        )
        .withColumn(
            "predicted_failure",
            when(col("prediction") == 1.0, "YES")
            .otherwise("NO")
        )
        .select(
            "machineID",
            "datetime",
            "volt",
            "rotate",
            "pressure",
            "vibration",
            "predicted_failure",
            "failure_probability",
            "risk_level"
        )
    )

    print("\nPREDICTION RESULTS")

    predictions.orderBy(
        col("failure_probability").desc()
    ).show(20, truncate=False)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    (
        predictions.coalesce(1)
        .write
        .mode("overwrite")
        .option("header", "true")
        .csv(str(OUTPUT_DIR))
    )

    print("Predictions saved to:", OUTPUT_DIR)
    print("Prediction pipeline completed successfully.")

except Exception as error:
    print("Prediction pipeline failed:", str(error))
    raise

finally:
    if spark is not None:
        try:
            spark.stop()
        except Exception:
            pass