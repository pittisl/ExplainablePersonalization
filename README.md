# On-device LLM Personalization via Explainable Model Selection

## Introduction

This is the code repository for our Mobisys paper: Never Start from Scratch: Expediting On-Device LLM Personalization via Explainable Model Selection. We proposed XPerT (eXplainale Personalized Tuning), a technique to accelerate on-device model personalization via starting from a well-matched model. 

This repository consists of three parts:

- 1 Code for LLM personalization.

- 2 An extension of our method to generative models of other modalities, specifically diffusion-based image generation models.

- 3 Code for fine-tuning an LLM on Android smartphones, including offline model format conversion and online fine-tuning.

## LLM personalization
Coming soon

## Selecting personalized diffusion model

### Requirements
* torch
* torchvision
* transformers
* diffusers
* datasets
* qwen_vl_utils
* accelerate

## General Usage
To run the code, you can either use our provided script `run.sh` or execute the file separately:

First use `personalized_ft.py` to finetune a personalized model:
```
accelerate launch personalized_ft.py \
  --pretrained_model_name_or_path=$BASE_MODEL \
  --train_data_dir=$DATASET \
  --caption_column="text" \
  --resolution=512 --center_crop --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --mixed_precision="bf16" \
  --max_train_steps=1500 \
  --checkpointing_steps=4000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" --lr_warmup_steps=0 \
  --output_dir=$FT_DIR \
```
`$DATASET` is the path of fine-tuning data which contains images and the corresponding captions. Make sure it can be loaded by `load_dataset()` function in `datasets`.

Then use `explain_diffusion.py` to find the explanation for the personalized image generation model in the previous step:
```
python3 explain_diffusion.py \
  --base_model=$BASE_MODEL \
  --personalized_model=$FT_DIR\
  --num_sample=50\
```


In `explain_diffusion.py`, the VLM used for summarizing differences is `Qwen/Qwen2.5-VL-7B-Instruct`. The text and image encoders are `openai/clip-vit-base-patch32`, and we use `nlphuji/mscoco_2014_5k_test_image_text_retrieval` to probe the personalized model's divergence. You can modify these settings in the corresponding code. 

After running the code, the explanation of the personalized model should be printed to the terminal

## Implementing LLM fine-tuning on smartphones
Coming soon
