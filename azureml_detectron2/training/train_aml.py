#!/usr/bin/env python
# coding: utf-8

# In[1]:


from azureml.core import Workspace

ws = Workspace.from_config()


# In[2]:


from azureml.core import Environment

detectron2 = Environment("detectron2")


# In[3]:


from azureml.core.conda_dependencies import CondaDependencies

conda_dep = CondaDependencies()
conda_dep.add_pip_package("joblib")

detectron2.python.conda_dependencies=conda_dep


# In[4]:


detectron2.docker.base_image = "esterakot/detectron2:latest"
detectron2.python.user_managed_dependencies = True
# detectron2.docker.base_image_registry.address
# detectron2.docker.base_image_registry.username
# detectron2.docker.base_image_registry.password


# In[5]:


from azureml.core.compute import ComputeTarget

cluster_name = "gpuforkeras"

compute_target = ComputeTarget(workspace=ws, name=cluster_name)


# In[6]:


from azureml.core import Dataset

detectron2_dataset = Dataset.get_by_name(ws, name='detectron2-demo')


# In[7]:


# detectron2_dataset.download()


# In[7]:


from azureml.core import ScriptRunConfig

src = ScriptRunConfig(source_directory='.',
                      script='train.py',
                      arguments=['--data-dir', detectron2_dataset.as_download()],
                      compute_target=compute_target,
                      environment=detectron2)


# In[8]:


from azureml.core import Experiment

run = Experiment(ws,'Detectron2-ballons').submit(src)
run.wait_for_completion(show_output=True)

