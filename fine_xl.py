import argparse
import logging
import math
import os
import random
import shutil
from pathlib import Path

import accelerate
import datasets
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer
from transformers.utils import ContextManagers

import diffusers
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel, PNDMScheduler
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel, compute_snr
from diffusers.utils import check_min_version, deprecate, is_wandb_available, make_image_grid
from diffusers.utils.import_utils import is_xformers_available
from diffusers.utils.torch_utils import is_compiled_module

from datasets import load_dataset
from PIL import Image
import requests
from transformers import AutoProcessor, CLIPVisionModel, CLIPTextModelWithProjection, CLIPVisionModelWithProjection
from transformers import AutoTokenizer, CLIPTextModel

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info

if is_wandb_available():
    import wandb

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--base_model", type=str, default="stabilityai/stable-diffusion-2-1-base", help=""
    )
    parser.add_argument(
        "--personalized_model", type=str, default="../sd15_van_2000", help=""
    )
    parser.add_argument(
        "--num_sample", type=int, default=2, help=""
    )
    parser.add_argument(
        "--e_ortho", type=float, default=1, help=""
    )
    parser.add_argument(
        "--e_decomp", type=float, default=0, help=""
    )
    args = parser.parse_args()

   
    return args

def main():
    args = parse_args()
    seed = 6666

    caption_data = load_dataset("nlphuji/mscoco_2014_5k_test_image_text_retrieval")
    
    pipe = StableDiffusionPipeline.from_pretrained(args.base_model, safety_checker=None, torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda")
    generator = torch.manual_seed(seed)

    for i in range (args.num_sample):
        prompts =  caption_data['test']['caption'][i][0]
        images = pipe(prompts, generator=generator).images
        image=images[0]
        image.save(f"base_gen/{i}.png")

    pipe = StableDiffusionPipeline.from_pretrained(args.personalized_model, safety_checker=None, torch_dtype=torch.bfloat16)
    pipe = pipe.to("cuda")
    generator = torch.manual_seed(seed)

    for i in range (args.num_sample):
        prompts =  caption_data['test']['caption'][i][0]
        images = pipe(prompts, generator=generator).images
        image=images[0]
        image.save(f"personalized_gen/{i}.png")

    clip_v_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
    clip_v_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    divergence = torch.zeros(1, 512)
    for i in range (args.num_sample):
        base_img=Image.open(f"base_gen/{i}.png")
        personalized_img=Image.open(f"personalized_gen/{i}.png")
        
        inputs = clip_v_processor(images=base_img, return_tensors="pt")
        outputs = clip_v_model(**inputs)
        #last_hidden_state = outputs.last_hidden_state
        base_emb = outputs.image_embeds

        inputs = clip_v_processor(images=personalized_img, return_tensors="pt")
        outputs = clip_v_model(**inputs)
        #last_hidden_state = outputs.last_hidden_state
        personalized_emb = outputs.image_embeds

        divergence=divergence+personalized_emb-base_emb

    summarization_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)
    summarization_processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

    word_list=[]
    for i in range (args.num_sample):
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"file://base_gen/{i}.png"},
                    {"type": "image", "image": f"file://personalized_gen/{i}.png"},
                    {"type": "text", "text": "Given the two image above, which image style is more characteristic of the second image than the first image? Reply with a list of single-word adjectives or simple phrases. (more than one)"},
                ],
            }
        ]

        text = summarization_processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = summarization_processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        
        # Inference
        generated_ids = summarization_model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = summarization_processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        keywords = output_text[0].split(', ')
        word_list = list(set(word_list).union(set(keywords)))

    sub_emb_list=[]
    clip_t_model = CLIPTextModelWithProjection.from_pretrained("openai/clip-vit-base-patch32")
    clip_t_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    for word in word_list:
        base_prompt=["a photo of "+sample[0] for sample in caption_data['test']['caption'][0:args.num_sample]]
        personalized_prompt=["a"+word+" photo of "+sample[0] for sample in caption_data['test']['caption'][0:args.num_sample]]
        
        inputs = clip_t_tokenizer(base_prompt, padding=True, return_tensors="pt")
        outputs = clip_t_model(**inputs)
        #last_hidden_state = outputs.last_hidden_state
        base_prompt_emb = outputs.text_embeds

        inputs = clip_t_tokenizer(personalized_prompt, padding=True, return_tensors="pt")
        outputs = clip_t_model(**inputs)
        #last_hidden_state = outputs.last_hidden_state
        personalized_prompt_emb = outputs.text_embeds
        divergence_prompt_emb=torch.mean(torch.stack([a - b for a, b in zip(personalized_prompt_emb, base_prompt_emb)]),dim=0)
        sub_emb_list.append(divergence_prompt_emb)

    divergence=divergence.squeeze()
    #print(sub_emb_list[0].shape)

    word_index=[]
    coef=[]
    ortho_emb=[]
    final_word_list=[]
    for i in range(len(sub_emb_list)):
        if i ==0:
            ortho_emb.append(sub_emb_list[i])
            word_index.append(i)
            final_word_list.append(word_list[i])
            proj=(torch.dot(divergence, sub_emb_list[i]) / torch.norm(sub_emb_list[i])) 
            coef.append(proj)
        else:
            orthogonality=[(torch.dot(emb, sub_emb_list[i]) / (torch.norm( emb)*torch.norm( sub_emb_list[i])))for emb in ortho_emb ]
            orthogonality=torch.sum(torch.stack(orthogonality))
            if orthogonality <args.e_ortho:
                word_index.append(i)
                ortho_emb.append(sub_emb_list[i])
                proj=(torch.dot(divergence, sub_emb_list[i]) / torch.norm(sub_emb_list[i])) 
                coef.append(proj)
                final_word_list.append(word_list[i])
        residual=[w*emb for w,emb in zip(coef,ortho_emb)]
        residual=torch.sum(torch.stack(residual))
        residual=torch.norm((divergence-residual), p=2)
        if residual<args.e_decomp:
            break
    print(final_word_list)
    print(coef)
        

    

if __name__ == "__main__":
    main()