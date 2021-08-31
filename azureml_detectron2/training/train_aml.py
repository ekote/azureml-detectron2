#!/usr/bin/env python
# coding: utf-8

from azureml.core.environment import Environment
# Create the environment
myenv = Environment(name="myenv")

# Set the container registry information
myenv.docker.base_image_registry.address = ""
myenv.docker.base_image_registry.username = ""
myenv.docker.base_image_registry.password = ""

myenv.inferencing_stack_version = "latest"  # This will install the inference specific apt packages.

# Define the packages needed by the model and scripts
from azureml.core.conda_dependencies import CondaDependencies
conda_dep = CondaDependencies()

# you must list azureml-defaults as a pip dependency
conda_dep.add_pip_package("azureml-defaults")
myenv.python.conda_dependencies=conda_dep

from azureml.core.model import InferenceConfig
# Use environment in InferenceConfig
inference_config = InferenceConfig(entry_script="train.py",
                                   environment=myenv)


import azureml.core
from azureml.core import Workspace

# Load the workspace
ws = Workspace.from_config()

from azureml.core.webservice import LocalWebservice, Webservice
from azureml.core import Model

deployment_config = LocalWebservice.deploy_configuration(port=8890)
# service = Model.deploy(ws, "myservice", [model], inference_config, deployment_config)
# service.wait_for_deployment(show_output = True)
# print(service.state)

from azureml.core import Workspace, Experiment, ScriptRunConfig, Dataset
from azureml.core.runconfig import MpiConfiguration
from azureml.core.compute import AmlCompute

cluster = AmlCompute(ws, "GPUengine") # Get existing cluster
# dataset = Dataset.get_by_name(workspace, "coco2017_trainval")

experiment = Experiment(ws, "Detectron2")

#dist_config = MpiConfiguration(node_count=4, process_count_per_node=8)

jobconfig = ScriptRunConfig(
  source_directory=".",
  script="train.py",
#   arguments=["--dataset", dataset.as_mount()],
  compute_target=cluster,
  environment=myenv
#  distributed_job_config=dist_config
  )

experiment.submit(jobconfig)

