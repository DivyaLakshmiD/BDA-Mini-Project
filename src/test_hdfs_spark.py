
from pyspark.sql import SparkSession
from pyspark.sql.functions import col


spark = SparkSession.builder \
    .appName("PredictiveMaintenance-HDFS-Test") \
    .master("local[*]") \
    .config("spark.hadoop.dfs.client.use.datanode.hostname", "true") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
hadoop_conf.set("dfs.client.use.datanode.hostname", "true")

print(
    "DataNode hostname setting:",
    hadoop_conf.get("dfs.client.use.datanode.hostname")
)

hdfs_path = "hdfs://localhost:9000/bda_predictive_maintenance/processed/maintenance_features.csv"

try:
    df = spark.read.option("header", True).option("inferSchema", True).csv(hdfs_path)

    print("\n========== HDFS + SPARK INTEGRATION ==========")
    print("Dataset loaded successfully from HDFS")
    print("Total columns:", len(df.columns))
    print("Column names:", df.columns)
    print("Sample records:")
    df.select("machineID", "datetime", "volt", "vibration", "failure_within_24h").show(5, truncate=False)

    print("Failure label counts:")
    df.groupBy("failure_within_24h").count().show()

    print("HDFS + Spark integration test completed successfully")

except Exception as e:
    print("HDFS + Spark integration test failed")
    print(str(e))
    raise

finally:
    spark.stop()