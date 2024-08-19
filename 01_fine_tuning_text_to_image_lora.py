# Databricks notebook source
# MAGIC %md
# MAGIC # Fine-tune Stable Diffusion XL with LoRA on Databricks
# MAGIC For fine-tuning, we use [Text-to-Image](https://huggingface.co/docs/diffusers/en/training/text2image), which is a technique to update the weights of a pre-trained text-to-image model. We use the [Diffusers](https://huggingface.co/docs/diffusers/en/index)' implementation of DreamBooth in this solution accelerator.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 0. Setup Environment
# MAGIC
# MAGIC Databricks Runtime version: v14.3 LTS ML GPU
# MAGIC
# MAGIC Node: Azure Standard_NC48ads_A100_v4 (A100 x 2)

# COMMAND ----------

# MAGIC %md
# MAGIC First, let's install related libraries and load some utility functions by running the following notebook.

# COMMAND ----------

# DBTITLE 1,Install requirements and load helper functions
# MAGIC %run ./99_utils

# COMMAND ----------

# MAGIC %md
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Generate an image with the normal version of Stable Diffusion XL before Fine-tuning 
# MAGIC Test the normal version of Stable Diffusion XL model before Fine-tuning

# COMMAND ----------

from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained(
  "stabilityai/stable-diffusion-xl-base-1.0", 
  torch_dtype=torch.float16).to("cuda")

image = pipeline(
  prompt="Leonardo da Vinci with a hoodie under the blue sky", 
  negative_prompt="ugly, deformed, disfigured, poor details, bad anatomy, bad face, bad finger", 
  num_inference_steps=25, 
  guidance_scale=7.5).images[0]
show_image(image)

# COMMAND ----------

# MAGIC %md
# MAGIC A good image is likely generated with this version of the Stable Diffusion XL model. However, from now on, we’ll update the model to generate images in a Naruto-style.

# COMMAND ----------

# MAGIC %md
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Prepare image dataset for fine-tuning
# MAGIC In this sample, we use the dataset called [naruto-blip-captions](https://huggingface.co/datasets/lambdalabs/naruto-blip-captions) which is disclosed in HuggingFace's repogitory. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2-1. Prepare Databricks Unity Catalog Volumes for storing image dataset

# COMMAND ----------

catalog = "hiroshi" # Name of the catalog we use to manage our assets (e.g. images, weights, datasets)
theme = "naruto" 
volume = "dataset"

# COMMAND ----------

# Make sure that the catalog and the schema exist
_ = spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}") 
_ = spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{theme}") 
_ = spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{theme}.{volume}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2-2. Download the image dataset and check some images

# COMMAND ----------

from datasets import load_dataset
dataset = load_dataset("lambdalabs/naruto-blip-captions")

# COMMAND ----------

show_image_grid(dataset["train"][:25]["image"], 5, 5)

# COMMAND ----------

dataset["train"][:25]["text"]

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2-3. Save the dataset into Unity Catalog Volumes

# COMMAND ----------

dataset_volumes_dir = f"/Volumes/{catalog}/{theme}/{volume}" # Path to the directories in UC Volumes
dataset.save_to_disk(dataset_volumes_dir)

# COMMAND ----------

# MAGIC %md
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Fine-tuning a Stable Diffusion XL model

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3-1. Set up TensorBoard
# MAGIC [TensorBoard](https://www.tensorflow.org/tensorboard) is an open source monitoring solution for model training. It reads an event log and exposes the training metrics in near real-time on its dashboard, which helps gauge the status of fine-tuning without having to wait until it's done.
# MAGIC
# MAGIC Note that when you write the event log to DBFS, it won't show until the file is closed for writing, which is when the training is complete. This is not good for real time monitoring. So we suggest to write the event log out to the driver node and run your TensorBoard from there (see the cell below on how to do this). Files stored on the driver node may get removed when the cluster terminates or restarts. But when you are running the training on Databricks notebook, MLflow will automatically log your Tensorboard artifacts, and you will be able to recover them later. You can find the example of this below.
# MAGIC
# MAGIC **Databricks recommends using MLflow, the default MLOps tool on the Databricks Data Intelligence Platform, to track experiments. However, this code does not support MLflow as of today. Please expect an update to the code soon.**

# COMMAND ----------

import os
from tensorboard import notebook

logdir = "/databricks/driver/logdir/sdxl/" # Write event log to the driver node
# logdir = "/dbfs/tmp/logdir/sdxl/" # Write event log to DBFS. It won't show until the file is closed for writing.
notebook.start("--logdir {} --reload_multifile True".format(logdir))

# COMMAND ----------

# MAGIC %md 
# MAGIC ### 3-2. Set some variables

# COMMAND ----------

os.environ["DATASET_NAME"] = dataset_volumes_dir

adapter_volume = "adaptor"
os.environ["OUTPUT_DIR"] = f"/Volumes/{catalog}/{theme}/{adapter_volume}"

os.environ["LOGDIR"] = logdir

# Make sure that the volume exists
_ = spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{theme}.{adapter_volume}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3-3. Set parameters for model traning
# MAGIC To ensure we can use Fine-tuning with LoRA on a heavy pipeline like Stable Diffusion XL, we use the following hyperparameters:
# MAGIC
# MAGIC * Gradient checkpointing (`--gradient_accumulation_steps`)
# MAGIC * Max Gradient Normalization (`--max_grad_norm`)
# MAGIC * Mixed-precision training (`--mixed-precision="fp16"`)
# MAGIC * Some other parameters are defined in `yamls/zero2.yaml`
# MAGIC <br>
# MAGIC
# MAGIC Other parameters:
# MAGIC * Use `--output_dir` to specify your LoRA model repository name.
# MAGIC * Use `--caption_column` to specify name of the caption column in your dataset.
# MAGIC * Make sure to pass the right number of GPUs to the parameter `num_processes` in `yamls/zero2.yaml`: e.g. `num_processes` should be 8 for `g5.48xlarge`.
# MAGIC <br>
# MAGIC
# MAGIC
# MAGIC The following cell will run for about 15 minutes on a single node cluster with 8xA10GPU instances on the default training images. 

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3-4. Run a training job 

# COMMAND ----------

# MAGIC %md
# MAGIC #### Option 1. Multi-GPU on Single node
# MAGIC
# MAGIC Using HF accelerate is better for single node based multi GPU environment. Here, we use the **Azure Standard_NC48ads_A100_v4** which has 2 of NVIDIA A100 GPUs inside it.

# COMMAND ----------

# MAGIC %sh accelerate launch --config_file yamls/zero2.yaml script/train_text_to_image_lora_sdxl.py \
# MAGIC   --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
# MAGIC   --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
# MAGIC   --dataset_name=$DATASET_NAME \
# MAGIC   --dataloader_num_workers=8 \
# MAGIC   --resolution=512 \
# MAGIC   --center_crop \
# MAGIC   --random_flip \
# MAGIC   --train_batch_size=2 \
# MAGIC   --gradient_accumulation_steps=4 \
# MAGIC   --max_grad_norm=1 \
# MAGIC   --max_train_steps=7500 \
# MAGIC   --learning_rate=1e-04 \
# MAGIC   --lr_scheduler="cosine" \
# MAGIC   --lr_warmup_steps=0 \
# MAGIC   --mixed_precision="fp16" \
# MAGIC   --checkpointing_steps=3750 \
# MAGIC   --validation_prompt="A naruto with blue eyes." \
# MAGIC   --seed=1337 \
# MAGIC   --output_dir=$OUTPUT_DIR \
# MAGIC   --report_to="tensorboard" \
# MAGIC   --logging_dir=$LOGDIR

# COMMAND ----------

# MAGIC %md
# MAGIC #### Option 2. Multi-GPU on Multi node
# MAGIC
# MAGIC If you want to use multi-node based cluster which Databricks provides and run training processes on each worker node's GPUs, then using [TorchDistributor](https://docs.databricks.com/en/machine-learning/train-model/distributed-training/spark-pytorch-distributor.html) is better option.

# COMMAND ----------

import os
from pyspark.ml.torch.distributor import TorchDistributor
distributor = TorchDistributor(
    num_processes=2,
    local_mode=True,
    use_gpu=True)

distributor.run(
  'personalized_image_generation/train_text_to_image_lora_sdxl.py', 
  '--pretrained_model_name_or_path=stabilityai/stable-diffusion-xl-base-1.0',
  '--pretrained_vae_model_name_or_path=madebyollin/sdxl-vae-fp16-fix',
  f'--dataset_name={os.environ["DATASET_NAME"]}',
  '--dataloader_num_workers=8',
  '--resolution=512',
  '--center_crop',
  '--random_flip',
  '--max_grad_norm=1',
  f'--output_dir={os.environ["OUTPUT_DIR"]}',
  '--train_batch_size=2',
  '--gradient_accumulation_steps=4',
  '--learning_rate=1e-04',
  '--lr_scheduler=constant',
  '--lr_warmup_steps=0',
  '--max_train_steps=7500',
  # '--num_train_epochs=1',
  '--checkpointing_steps=3750',
  '--seed=1337',
  '--validation_prompt="A naruto with blue eyes"',
  '--report_to=tensorboard',
  f'--logging_dir={os.environ["LOGDIR"]}',
  '--mixed_precision=fp16'
)

# COMMAND ----------

# MAGIC %sh ls -ltrh $OUTPUT_DIR

# COMMAND ----------

# MAGIC %md
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Test inference
# MAGIC Lets apply the fine-tuned LoRA adapter to the model and generate some images!

# COMMAND ----------

from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained(
  "stabilityai/stable-diffusion-xl-base-1.0", 
  torch_dtype=torch.float16).to("cuda")

pipeline.load_lora_weights(os.environ["OUTPUT_DIR"], weight_name="pytorch_lora_weights.safetensors")

image = pipeline(
  prompt="Leonardo da Vinci with a hoodie under the blue sky", 
  negative_prompt="ugly, deformed, disfigured, poor details, bad anatomy, bad face, bad finger", 
  num_inference_steps=25, 
  guidance_scale=7.5).images[0]
show_image(image)

# COMMAND ----------

# MAGIC %md
# MAGIC Probably you can find that the generated image from fine-tuned model is quite different style from the original model. The image should be drawn more in Naruto-style even though same prompt is used.

# COMMAND ----------

# MAGIC %md
# MAGIC Let's have one more test inference. This time, we'll generate 5 famous celebrities images in Naruto-style.

# COMMAND ----------

import os
import glob

persons = ["Abraham Lincoln", "William Shakespeare", "Cleopatra", "Marie Curie", "Wolfgang Amadeus Mozart"]
num_imgs_to_preview = len(persons)
imgs = []
for person in persons:
    imgs.append(
        pipeline(
            prompt=f"A photo of {person} with a hoodie under the blue sky",
            negative_prompt="ugly, deformed, disfigured, poor details, bad anatomy, bad face, bad finger", 
            num_inference_steps=25,
        ).images[0]
    )
show_image_grid(imgs[:num_imgs_to_preview], 1, num_imgs_to_preview)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Log the model to MLflow

# COMMAND ----------

import mlflow
import torch

class sdxl_fine_tuned(mlflow.pyfunc.PythonModel):
    def __init__(self, model_name):
        self.model_name = model_name

    def load_context(self, context):
        """
        This method initializes the vae and the model
        using the specified model repository.
        """
        # Initialize tokenizer and language model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            self.model_name, 
            torch_dtype=torch.float16)
        self.pipe.load_lora_weights(context.artifacts["repository"])
        self.pipe = self.pipe.to(self.device)

    def predict(self, context, model_input):
        """
        This method generates output for the given input.
        """
        prompt = model_input["prompt"][0]
        negative_prompt = model_input["negative_prompt"][0]
        num_inference_steps = model_input.get("num_inference_steps", [25])[0]
        # Generate the image
        image = self.pipe(
            prompt=prompt, 
            negative_prompt=negative_prompt, 
            num_inference_steps=num_inference_steps
        ).images[0]
        # Convert the image to numpy array for returning as prediction
        image_np = np.array(image)
        return image_np


# COMMAND ----------

model_name = "stabilityai/stable-diffusion-xl-base-1.0"
output = f'{os.environ["OUTPUT_DIR"]}/pytorch_lora_weights.safetensors'

# COMMAND ----------

import numpy as np
import pandas as pd
import mlflow
from mlflow.models.signature import ModelSignature
from mlflow.types import DataType, Schema, ColSpec, TensorSpec
import transformers, bitsandbytes, accelerate, deepspeed, diffusers, xformers, peft

experiment_name = f"/Workspace/Users/{dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()}/sdxl_experiment"
mlflow.set_experiment(experiment_name)

mlflow.set_registry_uri("databricks-uc")

# Define input and output schema
input_schema = Schema(
    [ColSpec(DataType.string, "prompt"), ColSpec(DataType.string, "negative_prompt"), ColSpec(DataType.long, "num_inference_steps")]
)
output_schema = Schema([TensorSpec(np.dtype(np.uint8), (-1, 768, 3))])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Define input example
input_example = pd.DataFrame(
    {
        "prompt": ["A photo of bill gates with a hoodie under the blue sky"], 
        "negative_prompt": ["ugly, deformed, disfigured, poor details, bad anatomy"], 
        "num_inference_steps": [25]
    }
)

# Log the model with its details such as artifacts, pip requirements and input example
with mlflow.start_run() as run:
    mlflow.pyfunc.log_model(
        "model",
        python_model=sdxl_fine_tuned(model_name),
        artifacts={"repository": output},
        pip_requirements=[
            "transformers==" + transformers.__version__,
            "bitsandbytes==" + bitsandbytes.__version__,
            "accelerate==" + accelerate.__version__,
            "deepspeed==" + deepspeed.__version__,
            "diffusers==" + diffusers.__version__,
            "xformers==" + xformers.__version__,
            "peft==" + peft.__version__,
        ],
        input_example=input_example,
        signature=signature,
    )
    mlflow.set_tag("dataset", dataset_volumes_dir)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Register the model to Unity Catalog

# COMMAND ----------

# Make sure that the schema for the model exist
_ = spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.model")

# Register the model 
registered_name = f"{catalog}.model.sdxl-fine-tuned-{theme}"
result = mlflow.register_model(
    "runs:/" + run.info.run_id + "/model",
    registered_name
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Load the registered model back to make inference
# MAGIC If you come accross an out of memory issue, restart the Python kernel to release the GPU memory occupied in Training. For this, uncomment and run the following cell, and re-define the variables such as ```theme```, ```catalog```, and ```volume_dir```.

# COMMAND ----------

#dbutils.library.restartPython()

# COMMAND ----------

def get_latest_model_version(mlflow_client, registered_name):
    latest_version = 1
    for mv in mlflow_client.search_model_versions(f"name='{registered_name}'"):
        version_int = int(mv.version)
        if version_int > latest_version:
            latest_version = version_int
    return latest_version


# COMMAND ----------

import mlflow
from mlflow import MlflowClient
import pandas as pd

mlflow.set_registry_uri("databricks-uc")
mlflow_client = MlflowClient()

registered_name = f"{catalog}.model.sdxl-fine-tuned-{theme}"
model_version = get_latest_model_version(mlflow_client, registered_name)
logged_model = f"models:/{registered_name}/{model_version}"

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# COMMAND ----------

# MAGIC %md
# MAGIC Armed with this model, the design team can now explore new variations of their products and even produce all-together new items reflective of the designs of previously produced items in their portfolio.

# COMMAND ----------

# Use any of the following token to generate personalized images: 'bcnchr', 'emslng', 'hsmnchr', 'rckchr', 'wdnchr'
input_example = pd.DataFrame(
    {
        "prompt": ["A photo of bill gates with a hoodie under the blue sky"], 
        "negative_prompt": ["ugly, deformed, disfigured, poor details, bad anatomy"], 
        "num_inference_steps": [25],
    }
)
image = loaded_model.predict(input_example)
show_image(image)

# COMMAND ----------

# Assign an alias to the model
mlflow_client.set_registered_model_alias(registered_name, "champion", model_version)

# COMMAND ----------

# MAGIC %md
# MAGIC Let's free up some memory again.

# COMMAND ----------

import gc
gc.collect()
torch.cuda.empty_cache()

# COMMAND ----------

# MAGIC %md
# MAGIC © 2024 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License. All included or referenced third party libraries are subject to the licenses set forth below.
# MAGIC
# MAGIC | library                                | description             | license    | source                                              |
# MAGIC |----------------------------------------|-------------------------|------------|-----------------------------------------------------|
# MAGIC | bitsandbytes | Accessible large language models via k-bit quantization for PyTorch. | MIT | https://pypi.org/project/bitsandbytes/
# MAGIC | diffusers | A library for pretrained diffusion models for generating images, audio, etc. | Apache 2.0 | https://pypi.org/project/diffusers/
# MAGIC | stable-diffusion-xl-base-1.0 | A model that can be used to generate and modify images based on text prompts. | CreativeML Open RAIL++-M License | https://github.com/Stability-AI/generative-models
