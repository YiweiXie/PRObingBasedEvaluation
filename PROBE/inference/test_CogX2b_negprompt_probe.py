"""
This script demonstrates how to generate a video using the CogVideoX model with the Hugging Face `diffusers` pipeline.
The script supports different types of video generation, including text-to-video (t2v), image-to-video (i2v),
and video-to-video (v2v), depending on the input data and different weight.

- text-to-video: THUDM/CogVideoX-5b or THUDM/CogVideoX-2b
- video-to-video: THUDM/CogVideoX-5b or THUDM/CogVideoX-2b
- image-to-video: THUDM/CogVideoX-5b-I2V

Running the Script:
To run the script, use the following command with appropriate arguments:

```bash
$ python cli_demo.py --prompt "A girl riding a bike." --model_path THUDM/CogVideoX-5b --generate_type "t2v"
```

Additional options are available to specify the model path, guidance scale, number of inference steps, video generation type, and output paths.
"""
from typing import List
import argparse
from typing import Literal
import pandas as pd

import torch
from diffusers import (
    CogVideoXPipeline,
    CogVideoXDDIMScheduler,
    CogVideoXDPMScheduler,
    CogVideoXImageToVideoPipeline,
    CogVideoXVideoToVideoPipeline,
)
import sys

sys.path.append('../T2VUnlearning_main/diffusers')
sys.path.append('../T2VUnlearning_main/receler')

from diffusers.utils import export_to_video, load_image, load_video
import os
import json

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def generate_video(
    prompt: str,
    model_path: str,
    eraser_path: str = None,
    eraser_rank: int = 128,
    output_path: str = "./output",
    image_or_video_path: str = "",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_frames: int = 49,
    dtype: torch.dtype = torch.bfloat16,
    generate_type: str = Literal["t2v", "i2v", "v2v"],  # i2v: image to video, v2v: video to video
    seed: int = 42,
    generate_clean: bool = False,
    embed_path: str = "",
    generate_neg: bool = False,
    generate_probe: bool = False,
):
    """
    Generates a video based on the given prompt and saves it to the specified path.

    Parameters:
    - prompt (str): The description of the video to be generated.
    - model_path (str): The path of the pre-trained model to be used.
    - eraser_path (str): The path of the eraser weights to be used.
    - eraser_rank (int): The rank of the eraser weights.
    - output_path (str): The path where the generated video will be saved.
    - num_inference_steps (int): Number of steps for the inference process. More steps can result in better quality.
    - guidance_scale (float): The scale for classifier-free guidance. Higher values can lead to better alignment with the prompt.
    - num_frames (int): Number of generated frames.
    - dtype (torch.dtype): The data type for computation (default is torch.bfloat16).
    - generate_type (str): The type of video generation (e.g., 't2v', 'i2v', 'v2v').·
    - seed (int): The seed for reproducibility.
    """

    # 1.  Load the pre-trained CogVideoX pipeline with the specified precision (bfloat16).
    # add device_map="balanced" in the from_pretrained function and remove the enable_model_cpu_offload()
    # function to use Multi GPUs.

    image = None
    video = None

    if generate_type == "i2v":
        pipe = CogVideoXImageToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype)
        image = load_image(image=image_or_video_path)
    elif generate_type == "t2v":
        pipe_ori = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype)
        pipe_CI = CogVideoXPipeline.from_pretrained(model_path, torch_dtype=dtype)
    else:
        pipe = CogVideoXVideoToVideoPipeline.from_pretrained(model_path, torch_dtype=dtype)
        video = load_video(image_or_video_path)

    # 2. Set Scheduler.
    # Can be changed to `CogVideoXDPMScheduler` or `CogVideoXDDIMScheduler`.
    # We recommend using `CogVideoXDDIMScheduler` for CogVideoX-2B.
    # using `CogVideoXDPMScheduler` for CogVideoX-5B / CogVideoX-5B-I2V.

    if generate_clean:
        pipe_ori.scheduler = CogVideoXDDIMScheduler.from_config(pipe_ori.scheduler.config, timestep_spacing="trailing")
        pipe_ori.to("cuda")
    if generate_probe:
        pipe_CI.scheduler = CogVideoXDDIMScheduler.from_config(pipe_CI.scheduler.config, timestep_spacing="trailing")
        pipe_CI.to("cuda")
    # 3. Enable CPU offload for the model.
    # turn off if you have multiple GPUs or enough GPU memory(such as H100) and it will cost less time in inference
    # and enable to("cuda")

    embed_path = embed_path
    embeds = torch.load(embed_path, map_location="cuda")  # e.g., {'<nudity>': tensor([5, 4096])}

    token, embedding_matrix = list(embeds.items())[0]  # token = "<nudity>", embedding_matrix = [5, 4096]

    num_vectors = embedding_matrix.shape[0]
    tokens = [token] + [f"{token}_{i}" for i in range(1, num_vectors)]  # eg: ['<nudity>', '<nudity_1>', ..., '<nudity_4>']

    pipe_CI.tokenizer.add_tokens(tokens, special_tokens=True)
    pipe_CI.text_encoder.resize_token_embeddings(len(pipe_CI.tokenizer))

    with torch.no_grad():
        for i, tok in enumerate(tokens):
            tok_id = pipe_CI.tokenizer.convert_tokens_to_ids(tok)
            pipe_CI.text_encoder.get_input_embeddings().weight[tok_id] = embedding_matrix[i]

    def make_soft_prompt(base_token="<nudity>", num_vectors=5):
        return " ".join([base_token] + [f"{base_token}_{i}" for i in range(1, num_vectors)])
    soft_prompt = make_soft_prompt(token, num_vectors)


    # 4. Generate the video frames based on the prompt.
    # `num_frames` is the Number of frames to generate.
    # This is the default value for 6 seconds video and 8 fps and will plus 1 frame for the first frame and 49 frames.

    for idx, row in df.iterrows():
        print("idx:", idx)
        if generate_clean:
            video_ori = pipe_ori(
                prompt=row['prompt'],  
                num_videos_per_prompt=1,
                num_inference_steps=num_inference_steps,
                num_frames=num_frames, 
                use_dynamic_cfg=True,
                guidance_scale=guidance_scale,
                generator=torch.Generator().manual_seed(seed),
            ).frames[0]
            for i in range(num_frames):
                video_ori[i].save(f"{output_path}/{idx}_{i}_ori.png")
        if generate_neg:
            video_neg = pipe_ori(
                prompt=row['prompt'],  
                negative_prompt = args.neg_prompt,
                num_videos_per_prompt=1,
                num_inference_steps=num_inference_steps,
                num_frames=num_frames, 
                use_dynamic_cfg=True,
                guidance_scale=guidance_scale,
                generator=torch.Generator().manual_seed(seed),
            ).frames[0]
            for i in range(num_frames):
                video_neg[i].save(f"{output_path}/{idx}_{i}_neg.png")
        if generate_probe:
            video_probe = pipe_CI(
                prompt=f"{soft_prompt} {row['prompt']}",
                negative_prompt = args.neg_prompt,
                num_videos_per_prompt=1,
                num_inference_steps=num_inference_steps,
                num_frames=num_frames, 
                use_dynamic_cfg=True,
                guidance_scale=guidance_scale,
                generator=torch.Generator().manual_seed(seed),
            ).frames[0]
            for i in range(num_frames):
                video_probe[i].save(f"{output_path}/{idx}_{i}_probe.png")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using CogVideoX")
    parser.add_argument("--prompt", type=str, required=True, help="The description of the video to be generated")
    parser.add_argument(
        "--image_or_video_path",
        type=str,
        default=None,
        help="The path of the image to be used as the background of the video",
    )
    parser.add_argument(
        "--model_path", type=str, default="THUDM/CogVideoX-5b", help="The path of the pre-trained model to be used"
    )
    parser.add_argument("--eraser_path", type=str, default=None, help="The path of the LoRA weights to be used")
    parser.add_argument("--eraser_rank", type=int, default=128, help="The rank of the LoRA weights")
    parser.add_argument(
        "--output_path", type=str, default="./output.mp4", help="The path where the generated video will be saved"
    )
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="The scale for classifier-free guidance")
    parser.add_argument(
        "--num_inference_steps", type=int, default=50, help="Number of steps for the inference process"
    )
    parser.add_argument("--num_frames", type=int, default=49, help="Number of frames to generate per prompt")
    parser.add_argument(
        "--generate_type", type=str, default="t2v", help="The type of video generation (e.g., 't2v', 'i2v', 'v2v')"
    )
    parser.add_argument(
        "--dtype", type=str, default="bfloat16", help="The data type for computation (e.g., 'float16' or 'bfloat16')"
    )
    parser.add_argument("--seed", type=int, default=42, help="The seed for reproducibility")
    parser.add_argument("--generate_clean", action="store_true", help="generate clean video for comparison")
    parser.add_argument("--generate_neg", action="store_true", help="generate clean video for comparison")
    parser.add_argument("--generate_probe", action="store_true", help="generate clean video for comparison")
    parser.add_argument("--neg_prompt", type=str, default="", help="neg_Concept")
    parser.add_argument("--embed_path", type=str, default="", help="embed_path")
    parser.add_argument("--csv_path", type=str, default="", help="csv_path")

    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16


    csv_path = args.csv_path  
    df = pd.read_csv(csv_path, encoding="latin1")

    generate_video(
            prompt=args.prompt,
            model_path=args.model_path,
            eraser_path=args.eraser_path,
            eraser_rank=args.eraser_rank,
            output_path=args.output_path,
            image_or_video_path=args.image_or_video_path,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            num_frames=args.num_frames,
            dtype=dtype,
            generate_type=args.generate_type,
            generate_clean=args.generate_clean,
            generate_neg=args.generate_neg,
            generate_probe=args.generate_probe,
            seed=args.seed,
            embed_path=args.embed_path,
    )
