
from pathlib import Path
import pandas as pd

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
from pyspark.ml import PipelineModel
from pyspark.ml.functions import vector_to_array


PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models" / "predictive_maintenance_rf"
FEATURES_FILE = PROJECT_DIR / "data" / "processed" / "maintenance_features.csv"


def predict_uploaded_csv(uploaded_file):
    spark = None

    try:
        uploaded_df = pd.read_csv(uploaded_file)
        uploaded_df.columns = uploaded_df.columns.str.strip()

        required_columns = [
            "datetime",
            "machineID",
            "volt",
            "rotate",
            "pressure",
            "vibration",
        ]

        missing = [
            name for name in required_columns
            if name not in uploaded_df.columns
        ]

        if missing:
            raise ValueError(
                "Missing required columns: " + ", ".join(missing)
            )

        uploaded_df["datetime"] = pd.to_datetime(
            uploaded_df["datetime"], errors="coerce"
        )
        uploaded_df["machineID"] = pd.to_numeric(
            uploaded_df["machineID"], errors="coerce"
        )

        for name in ["volt", "rotate", "pressure", "vibration"]:
            uploaded_df[name] = pd.to_numeric(
                uploaded_df[name], errors="coerce"
            )

        uploaded_df = uploaded_df.dropna(
            subset=required_columns
        ).copy()

        if uploaded_df.empty:
            raise ValueError(
                "No valid sensor records remain after checking the CSV."
            )

        if not FEATURES_FILE.exists():
            raise FileNotFoundError(
                f"Maintenance features not found: {FEATURES_FILE}"
            )

        if not MODEL_DIR.exists():
            raise FileNotFoundError(
                f"Saved model not found: {MODEL_DIR}"
            )

        spark = (
            SparkSession.builder
            .appName("BDA_Uploaded_CSV_Predictions")
            .master("local[*]")
            .getOrCreate()
        )
        spark.sparkContext.setLogLevel("WARN")

        
        features_pd = pd.read_csv(FEATURES_FILE)
        features_pd["datetime"] = pd.to_datetime(
            features_pd["datetime"], errors="coerce"
        )
        features_pd["machineID"] = pd.to_numeric(
            features_pd["machineID"], errors="coerce"
        )

        features_pd = features_pd.dropna(
            subset=["datetime", "machineID"]
        )

        features = spark.createDataFrame(features_pd)


        upload_spark = spark.createDataFrame(uploaded_df)
        upload_spark = upload_spark.withColumn(
            "datetime", col("datetime").cast("timestamp")
        )

        feature_columns = [
            "machineID",
            "datetime",
            "model",
            "age",
            "recent_error_count",
            "hours_since_maintenance",
        ]

        missing_features = [
            name for name in feature_columns
            if name not in features.columns
        ]

        if missing_features:
            raise ValueError(
                "Maintenance dataset is missing columns: "
                + ", ".join(missing_features)
            )

        joined = (
            upload_spark.alias("u")
            .join(
                features.select(*feature_columns).alias("f"),
                (
                    (col("u.machineID") == col("f.machineID"))
                    & (col("u.datetime") == col("f.datetime"))
                ),
                "left"
            )
        )

        prediction_input = joined.select(
            col("u.machineID").alias("machineID"),
            col("u.datetime").alias("datetime"),
            col("u.volt").alias("volt"),
            col("u.rotate").alias("rotate"),
            col("u.pressure").alias("pressure"),
            col("u.vibration").alias("vibration"),
            col("f.model").alias("model"),
            col("f.age").alias("age"),
            col("f.recent_error_count").alias("recent_error_count"),
            col("f.hours_since_maintenance").alias(
                "hours_since_maintenance"
            ),
        )

        prediction_input = prediction_input.dropDuplicates(
            ["machineID", "datetime"]
        )

        if prediction_input.filter(
            col("model").isNull()
            | col("age").isNull()
            | col("recent_error_count").isNull()
            | col("hours_since_maintenance").isNull()
        ).limit(1).count():
            raise ValueError(
                "Some uploaded machine/date records do not match the "
                "existing maintenance dataset. Use telemetry records "
                "whose machineID and datetime match that dataset."
            )

        model = PipelineModel.load(str(MODEL_DIR))

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
                "failure_probability",
                "risk_level",
                "predicted_failure",
            )
            .orderBy(col("failure_probability").desc())
        )

        return predictions.toPandas()

    finally:
        if spark is not None:
            spark.stop()
