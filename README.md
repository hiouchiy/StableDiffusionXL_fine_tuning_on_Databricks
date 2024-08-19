# Fine-tuning Stable Diffusion XL with LoRA on Databricks

This repository contains two Databricks notebooks designed to fine-tune and deploy the Stable Diffusion XL model using LoRA (Low-Rank Adaptation). The process is optimized for running on Databricks GPU instances and is split into the following notebooks:

- `01_fine_tuning_text_to_image_lora.py`: Covers steps from "2. Generating an Initial Image" to "7. Registering the Model to Unity Catalog".
- `02_deploy_model.py`: Focuses on "8. Model Serving with REST API".

## Table of Contents
1. [Setup Environment](#1-setup-environment)
2. [Generating an Initial Image](#2-generating-an-initial-image)
3. [Preparing the Dataset](#3-preparing-the-dataset)
4. [Fine-tuning the Model](#4-fine-tuning-the-model)
   - [TensorBoard and MLflow Integration](#41-tensorboard-and-mlflow-integration)
   - [Single-Node Fine-Tuning (Single GPU / Multi GPU)](#42-single-node-fine-tuning-single-gpu-multi-gpu)
   - [Multi-Node Fine-Tuning (Multi GPU)](#43-multi-node-fine-tuning-multi-gpu)
5. [Testing Inference](#5-testing-inference)
6. [Logging the Model to MLflow](#6-logging-the-model-to-mlflow)
7. [Registering the Model to Unity Catalog](#7-registering-the-model-to-unity-catalog)
8. [Model Serving with REST API](#8-model-serving-with-rest-api)
9. [Conclusion](#9-conclusion)
10. [License](#10-license)

## 1. Setup Environment

To start, the environment is configured to use Databricks Runtime version 14.3 LTS ML GPU, specifically on an Azure `Standard_NC48ads_A100_v4` instance with two NVIDIA A100 GPUs. The same setup can also be replicated on AWS by using an equivalent instance type, ensuring identical performance and results.

## 2. Generating an Initial Image

Before fine-tuning, we test the normal version of the Stable Diffusion XL model by generating an image. This helps establish a baseline for comparison after the model is fine-tuned.

## 3. Preparing the Dataset

We utilize the `naruto-blip-captions` dataset from HuggingFace to fine-tune the model. The dataset is prepared and stored in Unity Catalog Volumes for efficient access during training.

## 4. Fine-tuning the Model

The fine-tuning process involves several steps:

### 4.1 TensorBoard and MLflow Integration

In the current implementation, TensorBoard is used for monitoring the fine-tuning process because the original code was built with TensorBoard in mind. However, as a best practice, Databricks strongly recommends using [MLflow](https://mlflow.org/) for experiment tracking and model management. MLflow is natively integrated into the Databricks platform and provides more robust features for tracking experiments, managing models, and deploying machine learning workflows.

**Note:** We are currently in the process of updating the code to fully support MLflow. This will make the process of monitoring and managing experiments more seamless within the Databricks ecosystem.

### 4.2 Single-Node Fine-Tuning (Single GPU / Multi GPU)

For single-node configurations, where you are either using a single GPU or multiple GPUs on the same node, we use [Hugging Face's Accelerate](https://huggingface.co/docs/accelerate/index) as the execution engine. Accelerate simplifies the process of training models across different hardware setups, whether it’s on a single device or multiple GPUs. It abstracts away the complexity of distributed training and allows you to seamlessly switch between different configurations.

Example usage:
```python
!accelerate launch --config_file yamls/zero2.yaml script/train_text_to_image_lora_sdxl.py \
   --pretrained_model_name_or_path="stabilityai/stable-diffusion-xl-base-1.0" \
   --pretrained_vae_model_name_or_path="madebyollin/sdxl-vae-fp16-fix" \
   --dataset_name=$DATASET_NAME \
   --dataloader_num_workers=8 \
   --resolution=512 \
   --center_crop \
   --random_flip \
   --train_batch_size=2 \
   --gradient_accumulation_steps=4 \
   --max_grad_norm=1 \
   --max_train_steps=7500 \
   --learning_rate=1e-04 \
   --lr_scheduler="cosine" \
   --lr_warmup_steps=0 \
   --mixed_precision="fp16" \
   --checkpointing_steps=3750 \
   --validation_prompt="A naruto with blue eyes." \
   --seed=1337 \
   --output_dir=$OUTPUT_DIR \
   --report_to="tensorboard" \
   --logging_dir=$LOGDIR
```

### 4.3 Multi-Node Fine-Tuning (Multi GPU)

For multi-node configurations, where the training job runs across multiple nodes, each with multiple GPUs, we use [TorchDistributor](https://docs.databricks.com/en/machine-learning/train-model/distributed-training/spark-pytorch-distributor.html) as the execution engine. TorchDistributor is a powerful tool that enables distributed training with PyTorch across multiple nodes, leveraging the full power of Databricks clusters. It ensures that the training is efficiently parallelized across all available resources.

Example usage:
```python
from pyspark.ml.torch.distributor import TorchDistributor

distributor = TorchDistributor(
    num_processes=2,
    local_mode=True,
    use_gpu=True
)

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
  '--checkpointing_steps=3750',
  '--seed=1337',
  '--validation_prompt="A naruto with blue eyes"',
  '--report_to=tensorboard',
  f'--logging_dir={os.environ["LOGDIR"]}',
  '--mixed_precision=fp16'
)
```

## 5. Testing Inference

After fine-tuning, we test the model by generating images of famous celebrities in a Naruto-style. The fine-tuned model should produce images with noticeable stylistic differences compared to the original model.

## 6. Logging the Model to MLflow

The fine-tuned model is logged to MLflow, including the trained weights, environment configuration, and an input example for later use. This ensures that the model can be easily retrieved and used for inference or further fine-tuning.

## 7. Registering the Model to Unity Catalog

The model is registered in Unity Catalog, allowing it to be used across different Databricks workspaces with proper access control. This step also includes setting an alias for easy version management.

## 8. Model Serving with REST API

The second notebook, `02_deploy_model.py`, focuses on creating a REST API endpoint for serving the fine-tuned model. It demonstrates how to:
- Create, update, and delete a model serving endpoint.
- Monitor the endpoint's status.
- Score the model using the REST API.

## 9. Conclusion

This repository provides a comprehensive guide for fine-tuning and deploying a Stable Diffusion model on Databricks. It is suitable for users looking to leverage Databricks' capabilities for large-scale model training and serving. The process described here was validated on Azure Databricks, but it can be similarly executed on AWS Databricks using equivalent GPU instances.

## 10. License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
