# Databricks notebook source
# MAGIC %md
# MAGIC # Using TF Model Zoo
# MAGIC Based on: https://www.tensorflow.org/tutorials/images/transfer_learning

# COMMAND ----------

# MAGIC # We want all the latest mlflow features
# MAGIC %pip install mlflow==2.9.2 pynvml

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import os
import tensorflow as tf
import mlflow

from pyspark.sql import Window
from pyspark.sql import functions as F

# COMMAND ----------

# Setup vars for mlflow
username = spark.sql("SELECT current_user()").first()['current_user()']
experiment_path = f'/Users/{username}/scaling-tf-vision'

# Download datafiles to a dbfs directory
data_dir = f'/Users/{username}/weather'
dbfs_data_dir = f'/dbfs{data_dir}'
dbutils.fs.mkdirs(data_dir)

# COMMAND ----------

zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True,
    cache_dir = dbfs_data_dir)
csv_path, _ = os.path.splitext(zip_path)

dbutils_csv_path = csv_path.replace('/dbfs', '', 1) 

# COMMAND ----------

# In databricks we use the Delta lake tables
# We will save the csv and use that from delta

spark.sql(f"CREATE CATALOG IF NOT EXISTS tf_loading")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS tf_loading.weather_data")

weather_table = spark.read.csv(dbutils_csv_path, header=True, inferSchema=True)
display(weather_table)

# COMMAND ----------

weather_table.mode('overwrite').saveAsTable('tf_loading.weather_data.jena_climate_2009_2016')

# COMMAND ----------

# Split and normalise
## for timeseries we don't wanna randomSplit we need to split by time
windowSpec = Window.orderBy("Date Time")
processed_table = (weather_table.orderBy('Date Time')
                   .withColumn("row_number", F.row_number().over(windowSpec)))

display(processed_table)

# COMMAND ----------

num_rows = processed_table.count()

train_index = int(num_rows*0.7)
val_index = int(num_rows*0.9)

train_df = processed_table.filter(f'row_number < {train_index}').drop('row_number')
val_df = processed_table.filter(f'row_number >= {train_index}' and f'row_number < {val_index}').drop('row_number')
test_df = processed_table.filter(f'row_number >= {val_index}').drop('row_number')

### To reuse some of the latter code from the tutorial we can switch to pandas on spark api 
# Pandas on spark allows for reusing the pandas api whilst retaining the distributed nature of spark

train_pdf = train_df.pandas_api()
val_pdf = val_df.pandas_api()
test_pdf = test_df.pandas_api()

# COMMAND ----------

train_mean = train_pdf.mean()
train_std = train_pdf.std()

# To refactor to work with PySpark API
train_pdf = (train_pdf - train_mean) / train_std
val_pdf = (val_pdf - train_mean) / train_std
test_pdf = (test_pdf - train_mean) / train_std

# COMMAND ----------

# TODO - Windowing
# TODO - Generator class
# TODO - to Dataset
# TODO - Model