from pathlib import Path
import csv
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when

from pyspark.ml import Pipeline
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator
)

project_dir = Path(__file__).resolve().parent.parent
input_file = project_dir / "data" / "processed" / "maintenance_features.csv"
model_dir = project_dir / "models" / "predictive_maintenance_rf"

spark = (
    SparkSession.builder
    .appName("PredictiveMaintenanceModelTraining")
    .master("local[*]")
    .getOrCreate()
)

spark.sparkContext.setLogLevel("ERROR")

data = spark.read.csv(
    str(input_file),
    header=True,
    inferSchema=True
)

data = data.withColumn(
    "datetime",
    col("datetime").cast("timestamp")
).withColumn(
    "failure_within_24h",
    col("failure_within_24h").cast("double")
)

data = data.filter(col("datetime").isNotNull())

print("\nDATASET OVERVIEW")
print("Total records:", data.count())
data.groupBy("failure_within_24h").count().show()

training_data = data.filter(
    col("datetime") < "2015-10-01 00:00:00"
)

testing_data = data.filter(
    col("datetime") >= "2015-10-01 00:00:00"
)

print("\nTRAINING RECORDS:", training_data.count())
print("TESTING RECORDS:", testing_data.count())

model_indexer = StringIndexer(
    inputCol="model",
    outputCol="model_index",
    handleInvalid="keep"
)

model_encoder = OneHotEncoder(
    inputCols=["model_index"],
    outputCols=["model_encoded"]
)

feature_columns = [
    "volt",
    "rotate",
    "pressure",
    "vibration",
    "age",
    "recent_error_count",
    "hours_since_maintenance",
    "model_encoded"
]

assembler = VectorAssembler(
    inputCols=feature_columns,
    outputCol="features",
    handleInvalid="keep"
)

classifier = RandomForestClassifier(
    labelCol="failure_within_24h",
    featuresCol="features",
    predictionCol="prediction",
    probabilityCol="probability",
    numTrees=50,
    maxDepth=8,
    seed=42
)

pipeline = Pipeline(stages=[
    model_indexer,
    model_encoder,
    assembler,
    classifier
])

print("\nTRAINING RANDOM FOREST MODEL...")

trained_model = pipeline.fit(training_data)

print("\nGENERATING TEST PREDICTIONS...")

predictions = trained_model.transform(testing_data).cache()

print("\nSAMPLE PREDICTIONS")
predictions.select(
    "machineID",
    "datetime",
    "failure_within_24h",
    "prediction",
    "probability"
).show(10, truncate=False)

tp = predictions.filter(
    (col("failure_within_24h") == 1) &
    (col("prediction") == 1.0)
).count()

tn = predictions.filter(
    (col("failure_within_24h") == 0) &
    (col("prediction") == 0.0)
).count()

fp = predictions.filter(
    (col("failure_within_24h") == 0) &
    (col("prediction") == 1.0)
).count()

fn = predictions.filter(
    (col("failure_within_24h") == 1) &
    (col("prediction") == 0.0)
).count()

precision = tp / (tp + fp) if tp + fp else 0.0
recall = tp / (tp + fn) if tp + fn else 0.0
f1 = (
    2 * precision * recall / (precision + recall)
    if precision + recall else 0.0
)

pr_evaluator = BinaryClassificationEvaluator(
    labelCol="failure_within_24h",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderPR"
)

roc_evaluator = BinaryClassificationEvaluator(
    labelCol="failure_within_24h",
    rawPredictionCol="rawPrediction",
    metricName="areaUnderROC"
)

pr_auc = pr_evaluator.evaluate(predictions)
roc_auc = roc_evaluator.evaluate(predictions)

accuracy_evaluator = MulticlassClassificationEvaluator(
    labelCol="failure_within_24h",
    predictionCol="prediction",
    metricName="accuracy"
)

accuracy = accuracy_evaluator.evaluate(predictions)

print("\nMODEL EVALUATION")
print("-------------------------")
print("True Positives:", tp)
print("True Negatives:", tn)
print("False Positives:", fp)
print("False Negatives:", fn)
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")
print(f"PR-AUC:    {pr_auc:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")

print("\nCONFUSION MATRIX")
print("                  Predicted No Failure | Predicted Failure")
print(f"Actual No Failure: {tn:18d} | {fp:17d}")
print(f"Actual Failure:    {fn:18d} | {tp:17d}")

print("\nSAVING TRAINED MODEL...")

model_dir.parent.mkdir(parents=True, exist_ok=True)

try:
    trained_model.write().overwrite().save(str(model_dir))
    print("Model saved successfully to:", model_dir)
except Exception as e:
    print("Model could not be saved:", str(e))

print("\nSAVING MODEL EVALUATION RESULTS...")

results_dir = project_dir / "results"
results_dir.mkdir(parents=True, exist_ok=True)

predictions_file = results_dir / "test_predictions.csv"

with open(predictions_file, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow([
        "machineID",
        "datetime",
        "failure_within_24h",
        "prediction",
        "probability"
    ])

    for row in predictions.select(
        "machineID",
        "datetime",
        "failure_within_24h",
        "prediction",
        "probability"
    ).toLocalIterator():
        writer.writerow([
            row["machineID"],
            str(row["datetime"]),
            row["failure_within_24h"],
            row["prediction"],
            str(row["probability"])
        ])

metrics_file = results_dir / "model_metrics.txt"

with open(metrics_file, "w", encoding="utf-8") as file:
    file.write("PREDICTIVE MAINTENANCE MODEL EVALUATION\n")
    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"Precision: {precision:.4f}\n")
    file.write(f"Recall: {recall:.4f}\n")
    file.write(f"F1-score: {f1:.4f}\n")
    file.write(f"PR-AUC: {pr_auc:.4f}\n")
    file.write(f"ROC-AUC: {roc_auc:.4f}\n")
    file.write(f"True Positives: {tp}\n")
    file.write(f"True Negatives: {tn}\n")
    file.write(f"False Positives: {fp}\n")
    file.write(f"False Negatives: {fn}\n")

print("Predictions saved to:", predictions_file)
print("Metrics saved to:", metrics_file)

predictions.unpersist()
spark.stop()

print("\nMODEL TRAINING AND EVALUATION COMPLETED")