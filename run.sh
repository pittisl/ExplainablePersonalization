export DATASET_NAME='synthetic_mono_cap'
export BASE_MODEL="stabilityai/stable-diffusion-2-1-base"
export FT_DIR="../synthetic_mix_1500"

accelerate launch personalized_ft.py \
  --pretrained_model_name_or_path=$BASE_MODEL \
  --train_data_dir=$DATASET_NAME \
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

python3 fine_xl.py \
  --base_model=$BASE_MODEL \
  --personalized_model=$FT_DIR\
  --num_sample=50\