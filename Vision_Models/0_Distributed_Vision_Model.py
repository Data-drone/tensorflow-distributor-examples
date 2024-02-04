# Databricks notebook source
# MAGIC %md
# MAGIC # Vision Models using TF Model Zoo
# MAGIC Based on: https://www.tensorflow.org/tutorials/images/transfer_learning

# COMMAND ----------

# MAGIC # We want all the latest mlflow features
# MAGIC %pip install mlflow==2.10.0 pynvml

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %run ../utils/mlflow_dist_utils
# COMMAND ----------

import os
import tensorflow as tf
import mlflow

# COMMAND ----------

# Setup MLFlow variables

# Download datafiles to a dbfs directory
username = spark.sql("SELECT current_user()").first()['current_user()']
experiment_path = f'/Users/{username}/scaling-tf-vision'

data_dir = f'/Users/{username}/cats_and_dogs'
dbfs_data_dir = f'/dbfs{data_dir}'
dbutils.fs.mkdirs(data_dir)

# COMMAND ----------

## We cache to dbfs directory instead so that the data is persisted
_URL = 'https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip'
path_to_zip = tf.keras.utils.get_file('cats_and_dogs.zip', origin=_URL, extract=True,
                                      cache_dir = dbfs_data_dir)

PATH = os.path.join(os.path.dirname(path_to_zip), 'cats_and_dogs_filtered')

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

print(f"""Path: {PATH}, 
train_dir: {train_dir}, 
val_dir: {validation_dir}""")

# COMMAND ----------

# Validation download
os.listdir(PATH)

# COMMAND ----------

# We will wrap up the training code into a function that we can distribute
def build_model(image_shape):

    # In this example we take a base model add some more layers and change some settings
    # then use that to make a final model for training
    # We will wrap that logic in this function

    base_model = tf.keras.applications.MobileNetV2(input_shape=image_shape,
                                                       include_top=False,
                                                       weights='imagenet')
    base_model.trainable = False
    
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.2),
    ])

    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()

    prediction_layer = tf.keras.layers.Dense(1)

    # Build keras model
    inputs = tf.keras.Input(shape=(160, 160, 3))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)

    model = tf.keras.Model(inputs, outputs)

    return model


def build_datasets(train_dir: str, validation_dir: str, batch_size, image_size):

    # in this function we will setup our datasets

    train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                            shuffle=True,
                                                            batch_size=batch_size,
                                                            image_size=image_size)
    
    validation_dataset = tf.keras.utils.image_dataset_from_directory(validation_dir,
                                                                 shuffle=True,
                                                                 batch_size=batch_size,
                                                                 image_size=image_size)
    
    val_batches = tf.data.experimental.cardinality(validation_dataset)
    test_dataset = validation_dataset.take(val_batches // 5)
    validation_dataset = validation_dataset.skip(val_batches // 5)

    AUTOTUNE = tf.data.AUTOTUNE

    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    return train_dataset, validation_dataset, test_dataset


def wrapped_train_loop(gpus_per_node:int, parent_run_id:str=None):

    mlflow.set_experiment(experiment_path)

    ### Temp Config Params

    BATCH_SIZE = 32 
    TOTAL_EPOCHS = 10
    IMG_SIZE = (160, 160)
    IMG_SHAPE = IMG_SIZE + (3,)

    base_learning_rate = 0.0001
    #train_dir
    #validation_dir

    ### 

    strategy = tf.distribute.MultiWorkerMirroredStrategy()

    is_chief, is_lead_process = detect_lead_process(strategy, gpus_per_node)

    active_run = create_mlflow_run(is_chief, is_lead_process, parent_run_id)

    # We need to change batching for train vs validation and test
    train_dataset, validation_dataset, test_dataset = build_datasets(train_dir, validation_dir,
                                                                     BATCH_SIZE, IMG_SIZE)

    dist_train = strategy.experimental_distribute_dataset(train_dataset)
    
    with strategy.scope():
        model = build_model(IMG_SHAPE)

        model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate/10),
              metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0, name='accuracy')])

    model.fit(dist_train,
              epochs = TOTAL_EPOCHS,
              validation_data = validation_dataset,
              steps_per_epoch  = 10) # TODO Write some code to work out actual steps needed
    
# COMMAND ----------
    
from spark_tensorflow_distributor import MirroredStrategyRunner

gpus_per_node = 1
num_nodes = 2
slots = gpus_per_node * num_nodes

local = False

mlflow.set_experiment(experiment_path)
with mlflow.start_run(run_name='multi-worker-run') as run:

    runner = MirroredStrategyRunner(num_slots=slots, local_mode=local, 
                                use_gpu=True, use_custom_strategy=True)

    runner.run(wrapped_train_loop, gpus_per_node=gpus_per_node, parent_run_id=run.info.run_id)