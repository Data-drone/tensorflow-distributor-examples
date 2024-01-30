# Databricks notebook source
# MAGIC %md
# MAGIC # Petfinder example
# MAGIC Based on: https://www.tensorflow.org/tutorials/structured_data/preprocessing_layers

# COMMAND ----------

# MAGIC %md # Setup and Load Data
# MAGIC no changes in this section

# COMMAND ----------

# DBTITLE 1,Import Libs
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers

# COMMAND ----------

# DBTITLE 1,Load Dataset
dataset_url = 'http://storage.googleapis.com/download.tensorflow.org/data/petfinder-mini.zip'
csv_file = 'datasets/petfinder-mini/petfinder-mini.csv'

tf.keras.utils.get_file('petfinder_mini.zip', dataset_url,
                        extract=True, cache_dir='.')
dataframe = pd.read_csv(csv_file)

# COMMAND ----------

dataframe.head()

# COMMAND ----------

# DBTITLE 1,Create a target variable
# In the original dataset, `'AdoptionSpeed'` of `4` indicates
# a pet was not adopted.
dataframe['target'] = np.where(dataframe['AdoptionSpeed']==4, 0, 1)

# Drop unused features.
dataframe = dataframe.drop(columns=['AdoptionSpeed', 'Description'])

# COMMAND ----------

# DBTITLE 1,Split the DataFrame into training, validation, and test sets
train, val, test = np.split(dataframe.sample(frac=1), [int(0.8*len(dataframe)), int(0.9*len(dataframe))])

print(len(train), 'training examples')
print(len(val), 'validation examples')
print(len(test), 'test examples')

# COMMAND ----------

# MAGIC %md # Functions for building the dataset
# MAGIC 

# COMMAND ----------

# DBTITLE 1,Create an input pipeline using tf.data
def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  df = dataframe.copy()
  labels = df.pop('target')
  df = {key: value.values[:,tf.newaxis] for key, value in dataframe.items()}
  ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  ds = ds.prefetch(batch_size)
  return ds

# COMMAND ----------

# DBTITLE 1,Apply the Keras preprocessing layers
def get_normalization_layer(name, dataset):
  # Create a Normalization layer for the feature.
  normalizer = layers.Normalization(axis=None)

  # Prepare a Dataset that only yields the feature.
  feature_ds = dataset.map(lambda x, y: x[name])

  # Learn the statistics of the data.
  normalizer.adapt(feature_ds)

  return normalizer

# COMMAND ----------

# DBTITLE 1,Categorical columns
def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
  # Create a layer that turns strings into integer indices.
  if dtype == 'string':
    index = layers.StringLookup(max_tokens=max_tokens)
  # Otherwise, create a layer that turns integer values into integer indices.
  else:
    index = layers.IntegerLookup(max_tokens=max_tokens)

  # Prepare a `tf.data.Dataset` that only yields the feature.
  feature_ds = dataset.map(lambda x, y: x[name])

  # Learn the set of possible values and assign them a fixed integer index.
  index.adapt(feature_ds)

  # Encode the integer indices.
  encoder = layers.CategoryEncoding(num_tokens=index.vocabulary_size())

  # Apply multi-hot encoding to the indices. The lambda function captures the
  # layer, so you can use them, or include them in the Keras Functional model later.
  return lambda feature: encoder(index(feature))

# COMMAND ----------

# MAGIC %md # Setting up training process for distribution

# COMMAND ----------

# building out training

def wrapped_train_loop():

    # issue with mirrored strategy see: https://github.com/tensorflow/tensorflow/issues/62234
    #strategy = tf.distribute.MirroredStrategy()
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
    communication= tf.distribute.experimental.CollectiveCommunication.RING
    )

    batch_size = 256

    with strategy.scope():
        train_ds = df_to_dataset(train, batch_size=batch_size)
        val_ds = df_to_dataset(val, shuffle=False, batch_size=batch_size)
        test_ds = df_to_dataset(test, shuffle=False, batch_size=batch_size)

        all_inputs = []
        encoded_features = []

        # Numerical features.
        for header in ['PhotoAmt', 'Fee']:
            numeric_col = tf.keras.Input(shape=(1,), name=header)
            normalization_layer = get_normalization_layer(header, train_ds)
            encoded_numeric_col = normalization_layer(numeric_col)
            all_inputs.append(numeric_col)
            encoded_features.append(encoded_numeric_col)

        age_col = tf.keras.Input(shape=(1,), name='Age', dtype='int64')

        encoding_layer = get_category_encoding_layer(name='Age',
                                                    dataset=train_ds,
                                                    dtype='int64',
                                                    max_tokens=5)
        encoded_age_col = encoding_layer(age_col)
        all_inputs.append(age_col)
        encoded_features.append(encoded_age_col)

        categorical_cols = ['Type', 'Color1', 'Color2', 'Gender', 'MaturitySize',
                            'FurLength', 'Vaccinated', 'Sterilized', 'Health', 'Breed1']

        for header in categorical_cols:
            categorical_col = tf.keras.Input(shape=(1,), name=header, dtype='string')
            encoding_layer = get_category_encoding_layer(name=header,
                                                        dataset=train_ds,
                                                        dtype='string',
                                                        max_tokens=5)
            encoded_categorical_col = encoding_layer(categorical_col)
            all_inputs.append(categorical_col)
            encoded_features.append(encoded_categorical_col)

        all_features = tf.keras.layers.concatenate(encoded_features)
        x = tf.keras.layers.Dense(32, activation="relu")(all_features)
        x = tf.keras.layers.Dropout(0.5)(x)
        output = tf.keras.layers.Dense(1)(x)

        model = tf.keras.Model(all_inputs, output)

        model.compile(optimizer='adam',
                    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                    metrics=["accuracy"])

        model.fit(train_ds, epochs=10, validation_data=val_ds)

# COMMAND ----------
        
from spark_tensorflow_distributor import MirroredStrategyRunner

num_gpus_per_node = 2
num_nodes = 1
slots = num_gpus_per_node * num_nodes
local = True if num_nodes == 1 else False 

# hyperparams (batch_size * slots)
runner = MirroredStrategyRunner(num_slots=slots, local_mode=local, 
                                use_gpu=True, use_custom_strategy=True)

runner.run(wrapped_train_loop)

# COMMAND ----------