# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # Tensorflow Logging and Monitoring
# MAGIC Based on: https://keras.io/examples/vision/sam/
# MAGIC
# MAGIC
# MAGIC We will now add mlflow

# COMMAND ----------

# MAGIC # We want all the latest mlflow features
# MAGIC %pip install mlflow==2.9.2 pynvml

# COMMAND ----------

from tensorflow import keras
from transformers import TFSamModel, SamProcessor
import numpy as np
import tensorflow as tf
from PIL import Image
import glob
import os

import mlflow

# COMMAND ----------

# Logging Setup
# Databricks configuration and MLflow setup
browser_host = spark.conf.get("spark.databricks.workspaceUrl")
db_host = f"https://{browser_host}"
db_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# MLflow configuration
username = spark.sql("SELECT current_user()").first()['current_user()']
experiment_path = f'/Users/{username}/scaling-tensorflow'

# Model Checkpoint Dir
checkpoint_dir = '/local_disk0/training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

# COMMAND ----------

# this downloads the data and sets it up on our node
remote_path = "https://hf.co/datasets/sayakpaul/sample-datasets/resolve/main/breast-cancer-dataset.tar.gz"
dataset_path = keras.utils.get_file(
    "breast-cancer-dataset.tar.gz", remote_path, untar=True
)

# COMMAND ----------
class Generator:
    """Generator class for processing the images and the masks for SAM fine-tuning."""

    def __init__(self, dataset_path, processor):
        self.dataset_path = dataset_path
        self.image_paths = sorted(
            glob.glob(os.path.join(self.dataset_path, "images/*.png"))
        )
        self.label_paths = sorted(
            glob.glob(os.path.join(self.dataset_path, "labels/*.png"))
        )
        self.processor = processor

    def __call__(self):
        for image_path, label_path in zip(self.image_paths, self.label_paths):
            image = np.array(Image.open(image_path))
            ground_truth_mask = np.array(Image.open(label_path))

            # get bounding box prompt
            prompt = self.get_bounding_box(ground_truth_mask)

            # prepare image and prompt for the model
            inputs = self.processor(image, input_boxes=[[prompt]], return_tensors="np")

            # remove batch dimension which the processor adds by default
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}

            # add ground truth segmentation
            inputs["ground_truth_mask"] = ground_truth_mask

            yield inputs

    def get_bounding_box(self, ground_truth_map):
        # get bounding box from mask
        y_indices, x_indices = np.where(ground_truth_map > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        # add perturbation to bounding box coordinates
        H, W = ground_truth_map.shape
        x_min = max(0, x_min - np.random.randint(0, 20))
        x_max = min(W, x_max + np.random.randint(0, 20))
        y_min = max(0, y_min - np.random.randint(0, 20))
        y_max = min(H, y_max + np.random.randint(0, 20))
        bbox = [x_min, y_min, x_max, y_max]

        return bbox

def dice_loss(y_true, y_pred, smooth=1e-5):
    y_pred = tf.sigmoid(y_pred)
    reduce_axis = list(range(2, len(y_pred.shape)))
    #if batch_size > 1:
        # reducing spatial dimensions and batch
    reduce_axis = [0] + reduce_axis
    intersection = tf.reduce_sum(y_true * y_pred, axis=reduce_axis)
    y_true_sq = tf.math.pow(y_true, 2)
    y_pred_sq = tf.math.pow(y_pred, 2)

    ground_o = tf.reduce_sum(y_true_sq, axis=reduce_axis)
    pred_o = tf.reduce_sum(y_pred_sq, axis=reduce_axis)
    denominator = ground_o + pred_o
    # calculate DICE coefficient
    loss = 1.0 - (2.0 * intersection + 1e-5) / (denominator + 1e-5)
    loss = tf.reduce_mean(loss)

    return loss

# COMMAND ----------

# MAGIC %md
# MAGIC # Train Loop
# MAGIC
# MAGIC For advanced training code, we need to wrap it in a train loop and make some changes
# MAGIC
# MAGIC Core Changes:
# MAGIC - 1) Add the Strategy Line - Here we choose Mirror
# MAGIC - 2) wrapping the model and optimizer in the strategy
# MAGIC - 3) Distribute the dataset
# MAGIC - 4) Wrap the original train loop with a distribution mechanism

# COMMAND ----------

# Logic we wrap in a training function

def wrapped_train_loop(global_batch_size:int=2):

    mlflow.set_experiment(experiment_path)
    active_run = mlflow.start_run(run_name='SAM Model', log_system_metrics=True)
    
    learning_rate = 1e-5

    mlflow.log_params({'learning_rate': learning_rate})

    # 1) Add Strategy
    strategy = tf.distribute.MirroredStrategy()

    # 2) Wrap the Model and Optimizer
    with strategy.scope():
        model = TFSamModel.from_pretrained("facebook/sam-vit-base")
        processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
        optimizer = keras.optimizers.Adam(learning_rate)

        # for saving model checkpoints
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)


    output_signature = {
        "pixel_values": tf.TensorSpec(shape=(3, None, None), dtype=tf.float32),
        "original_sizes": tf.TensorSpec(shape=(None,), dtype=tf.int64),
        "reshaped_input_sizes": tf.TensorSpec(shape=(None,), dtype=tf.int64),
        "input_boxes": tf.TensorSpec(shape=(None, None), dtype=tf.float64),
        "ground_truth_mask": tf.TensorSpec(shape=(None, None), dtype=tf.int32),
    }

    # Prepare the dataset object.
    train_dataset_gen = Generator(dataset_path, processor)
    train_ds = tf.data.Dataset.from_generator(
        train_dataset_gen, output_signature=output_signature
    )

    shuffle_buffer = 4

    train_ds = (
        train_ds.cache()
        .shuffle(shuffle_buffer)
        .batch(global_batch_size)
        #.prefetch(buffer_size=auto)
    )

    mlflow_dataset = mlflow.data.tensorflow_dataset.from_tensorflow(
        features=train_ds, name='breast-cancer-dataset'
    )

    mlflow.log_input(mlflow_dataset, context="training")

    # 3) Distribute the dataset
    dist_dataset = strategy.experimental_distribute_dataset(train_ds)
    
    # do we need to change this
    
    for layer in model.layers:
        if layer.name in ["vision_encoder", "prompt_encoder"]:
            layer.trainable = False


    #@tf.function
    def train_step(inputs):
        with tf.GradientTape() as tape:
            # pass inputs to SAM model
            outputs = model(
                pixel_values=inputs["pixel_values"],
                input_boxes=inputs["input_boxes"],
                multimask_output=False,
                training=True,
            )

            predicted_masks = tf.squeeze(outputs.pred_masks, 1)
            ground_truth_masks = tf.cast(inputs["ground_truth_mask"], tf.float32)

            # calculate loss over predicted and ground truth masks
            loss = dice_loss(tf.expand_dims(ground_truth_masks, 1), predicted_masks)
            # update trainable variables
            trainable_vars = model.trainable_variables
        grads = tape.gradient(loss, trainable_vars)
        optimizer.apply_gradients(zip(grads, trainable_vars))

        return loss
    
    # 4) Wrap the train function in a distribution mechanism
    ## The reduce step is to aggregate the losses from across the different shards
    @tf.function
    def distributed_train_step(dist_inputs):
        per_replica_losses = strategy.run(train_step, args=(dist_inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                         axis=None)

    
    for epoch in range(3):
        for step, inputs in enumerate(dist_dataset):
            loss = distributed_train_step(inputs)
            mlflow.log_metrics({'loss': loss}, step=step)
        print(f"Epoch {epoch + 1}: Loss = {loss}")

        checkpoint.save(checkpoint_prefix)

    mlflow.tensorflow.save_model(model, 'dbfs:/databricks/mlflow-tracking/4225188182853316')

    mlflow.end_run()

# COMMAND ----------
        
# MAGIC %md
# MAGIC # Execute the code
# MAGIC 
# MAGIC Now we can fire off the train function with the Mirror Strategy runner 

# COMMAND ----------
        
from spark_tensorflow_distributor import MirroredStrategyRunner

num_gpus_per_node = 2
num_nodes = 1
slots = num_gpus_per_node * num_nodes
local = True if num_nodes == 1 else False 

# hyperparams (batch_size * slots)
global_batch_size = 4

runner = MirroredStrategyRunner(num_slots=slots, local_mode=local, 
                                use_gpu=True, use_custom_strategy=True)

runner.run(wrapped_train_loop, global_batch_size=global_batch_size)

# COMMAND ----------

# Cleanup code

## Tensorflow doesn't seem to clean up after itself so we need use numba to do that
## Note that means that we need to detach and reattach too as numba breaks other things as well

from numba import cuda 

device_list = cuda.list_devices()

# the device_list returns an object so we need to use enumerate to get id
for device_id, device in enumerate(device_list):
    current_device = cuda.select_device(device_id)
    current_device.reset()