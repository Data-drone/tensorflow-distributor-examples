# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC #### Some helper scripts for logging distributed mlflow system metrics

# COMMAND ----------

import tensorflow as tf
from typing import Tuple
import mlflow
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID

# COMMAND ----------

def detect_lead_process(strategy: tf.distribute.Strategy, gpus_per_node: int) -> Tuple[bool, bool]:

    # In order to log system stats on all workers we need to
    # - make sure to run system stats logging only once per machine
    # - ensure that param logging just happens on the chief node
    

    task_id = strategy.cluster_resolver.task_id

    if task_id == 0:
        chief = True
        lead_process = True
        return chief, lead_process
    elif gpus_per_node:
        return False, True
    elif task_id & gpus_per_node == 0:
        return False, True
    else:
        return False, False 
    
# COMMAND ----------

def create_mlflow_run(is_chief: bool, is_lead: bool, parent_run_id: str) -> mlflow.entities.Run:

    """
    Starts a new MLflow run with specific configurations based on the role of the current process. 
    
    This function initializes an MLflow run, setting it as the active run. 
    If the process is identified as both chief and lead, it starts a run named 'chief_process' and enables 
    automatic logging of TensorFlow metrics. If the process is only a lead (not chief), 
    it starts a run named 'child_process' but disables TensorFlow autologging.
    
    Parameters:
    - is_chief (bool): A flag indicating if the current process is the chief process.
    - is_lead (bool): A flag indicating if the current process is a lead process. 
                        A lead process can be a chief or any process designated for special tasks.
    - parent_run_id (str): The ID of the parent MLflow run. This allows the new run to be nested or linked to an existing run.
    
    Returns:
    - mlflow.entities.Run: An object representing the newly created MLflow run. 
                            This object contains details about the run, including its ID, status, and metadata.
    
    Raises:
    - Exception: If MLflow is not properly configured or if there's an error in starting the run, an exception will be raised by the MLflow library.
    
    Example:
    >>> run = create_mlflow_run(is_chief=True, is_lead=True, parent_run_id='12345')
    >>> print(run.info.run_id)
    """
   
    if is_chief and is_lead:
        active_run = mlflow.start_run(run_name='chief_process', 
                                      log_system_metrics=True,
                                      tags = {MLFLOW_PARENT_RUN_ID: parent_run_id})
        mlflow.tensorflow.autolog()

    elif is_lead:
        active_run = mlflow.start_run(run_name='child_process', 
                                      log_system_metrics=True,
                                      tags = {MLFLOW_PARENT_RUN_ID: parent_run_id})
        mlflow.tensorflow.autolog(disable=True)

    return active_run