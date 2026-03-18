"""Code adapted from: https://github.com/huggingface/diffusers/tree/main/examples/textual_inversion"""

import argparse
import logging
import math
import os
import random
import warnings
from pathlib import Path
import json
import open_clip
import glob
import numpy as np
import PIL
import re
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder
from torch.utils.checkpoint import checkpoint

# TODO: remove and import from diffusers.utils when the new version of diffusers is released
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor
from diffusers.utils import export_to_video, load_image, load_video

import sys
sys.path.append('../T2VUnlearning_main/diffusers')
sys.path.append('../T2VUnlearning_main/receler')

import diffusers
from diffusers import (CogVideoXDDIMScheduler,
                       CogVideoXDPMScheduler,
                       CogVideoXImageToVideoPipeline,
                       CogVideoXVideoToVideoPipeline)
from ..T2VUnlearning_main.safree_cogvideo_pipeline import CogVideoXPipeline, projection_matrix, mask_to_onp
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
import torch.nn as nn

if is_wandb_available():
    import wandb

if version.parse(version.parse(PIL.__version__).base_version) >= version.parse("9.1.0"):
    PIL_INTERPOLATION = {
        "linear": PIL.Image.Resampling.BILINEAR,
        "bilinear": PIL.Image.Resampling.BILINEAR,
        "bicubic": PIL.Image.Resampling.BICUBIC,
        "lanczos": PIL.Image.Resampling.LANCZOS,
        "nearest": PIL.Image.Resampling.NEAREST,
    }
else:
    PIL_INTERPOLATION = {
        "linear": PIL.Image.LINEAR,
        "bilinear": PIL.Image.BILINEAR,
        "bicubic": PIL.Image.BICUBIC,
        "lanczos": PIL.Image.LANCZOS,
        "nearest": PIL.Image.NEAREST,
    }
# ------------------------------------------------------------------------------


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.17.0.dev0")

logger = get_logger(__name__)


def save_model_card(repo_id: str, images=None, base_model=str, repo_folder=None):
    img_str = ""
    for i, image in enumerate(images):
        image.save(os.path.join(repo_folder, f"image_{i}.png"))
        img_str += f"![img_{i}](./image_{i}.png)\n"

    yaml = f"""
---
license: creativeml-openrail-m
base_model: {base_model}
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- diffusers
- textual_inversion
inference: true
---
    """
    model_card = f"""
# Textual inversion text2image fine-tuning - {repo_id}
These are textual inversion adaption weights for {base_model}. You can find some example images in the following. \n
{img_str}
"""
    with open(os.path.join(repo_folder, "README.md"), "w") as f:
        f.write(yaml + model_card)


def log_validation(text_encoder, tokenizer, transformer, vae, args, accelerator, weight_dtype, epoch):
    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images with prompt:"
        f" {args.validation_prompt}."
    )
    # create pipeline (note: unet and vae are loaded again in float32)
    pipeline = CogVideoXPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        text_encoder=accelerator.unwrap_model(text_encoder),
        tokenizer=tokenizer,
        transformer=transformer,
        vae=vae,
        revision=args.revision,
        torch_dtype=weight_dtype,
    )
    pipeline.scheduler = CogVideoXDDIMScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    # run inference
    generator = None if args.seed is None else torch.Generator(device=accelerator.device).manual_seed(args.seed)
    images = []
    for _ in range(args.num_validation_images):
        with torch.autocast("cuda"):
            image = pipeline(args.validation_prompt, num_inference_steps=50, generator=generator,
                                    num_videos_per_prompt=1,
                                    num_frames=1,
                                    use_dynamic_cfg=True,
                                    guidance_scale=3.5).images[0]
        images.append(image)

    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            np_images = np.stack([np.asarray(img) for img in images])
            tracker.writer.add_images("validation", np_images, epoch, dataformats="NHWC")
        if tracker.name == "wandb":
            tracker.log(
                {
                    "validation": [
                        wandb.Image(image, caption=f"{i}: {args.validation_prompt}") for i, image in enumerate(images)
                    ]
                }
            )

    del pipeline
    torch.cuda.empty_cache()
    return images


def save_progress(text_encoder, placeholder_token_ids, accelerator, args, save_path):
    logger.info("Saving embeddings")
    learned_embeds = (
        accelerator.unwrap_model(text_encoder)
        .get_input_embeddings()
        .weight[min(placeholder_token_ids) : max(placeholder_token_ids) + 1]
    )
    learned_embeds_dict = {args.placeholder_token: learned_embeds.detach().cpu()}
    torch.save(learned_embeds_dict, save_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--save_steps",
        type=int,
        default=500,
        help="Save learned_embeds.bin every X updates steps.",
    )
    parser.add_argument(
        "--save_as_full_pipeline",
        action="store_true",
        help="Save the complete stable diffusion pipeline.",
    )
    parser.add_argument(
        "--num_vectors",
        type=int,
        default=1,
        help="How many textual inversion vectors shall be used to learn the concept.",
    )
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--train_data_dir", type=str, default=None, required=True, help="A folder containing the training data."
    )
    parser.add_argument(
        "--placeholder_token",
        type=str,
        default=None,
        required=True,
        help="A token to use as a placeholder for the concept.",
    )
    parser.add_argument(
        "--initializer_token", type=str, default=None, required=True, help="A token to use as initializer word."
    )
    parser.add_argument("--learnable_property", type=str, default="object", help="Choose between 'object' and 'style'")
    parser.add_argument("--repeats", type=int, default=100, help="How many times to repeat the training data.")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution."
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=5000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        help="A prompt that is used during validation to verify that the model is learning.",
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=4,
        help="Number of images that should be generated during validation with `validation_prompt`.",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=100,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--validation_epochs",
        type=int,
        default=None,
        help=(
            "Deprecated in favor of validation_steps. Run validation every X epochs. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming"
            " training using `--resume_from_checkpoint`."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=(
            "Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`."
            " See Accelerator::save_state https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.save_state"
            " for more docs"
        ),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    
    #esd arguments
    parser.add_argument("--esd_checkpoint", type=str, default="", help="checkpoint for esd")

    parser.add_argument("--neg_prompt", type=str, default="", help="negative prompt for safree")

    #number of images in training set
    parser.add_argument('--num_train_images', type=int, default=1000, help='number of images in training set')

    #i2p arguments
    parser.add_argument('--i2p', action='store_true', help='i2p dataset')
    parser.add_argument('--i2p_metadata_path', type=str, default="", help='i2p metadata path')

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.train_data_dir is None:
        raise ValueError("You must specify a train data directory.")

    return args


video_templates_small = [
    "a video of a {}",
    "a rendering of a {}",
    "a cropped video of the {}",
    "the video of a {}",
    "a video of a clean {}",
    "a video of a dirty {}",
    "a dark video of the {}",
    "a video of my {}",
    "a video of the cool {}",
    "a close-up video of a {}",
    "a bright video of the {}",
    "a cropped video of a {}",
    "a video of the {}",
    "a good video of the {}",
    "a video of one {}",
    "a close-up video of the {}",
    "a rendition of the {}",
    "a video of the clean {}",
    "a rendition of a {}",
    "a video of a nice {}",
    "a good video of a {}",
    "a video of the nice {}",
    "a video of the small {}",
    "a video of the weird {}",
    "a video of the large {}",
    "a video of a cool {}",
    "a video of a small {}",
]

video_style_templates_small = [
    "a painting in the style of {}",
    "a rendering in the style of {}",
    "a cropped painting in the style of {}",
    "the painting in the style of {}",
    "a clean painting in the style of {}",
    "a dirty painting in the style of {}",
    "a dark painting in the style of {}",
    "a picture in the style of {}",
    "a cool painting in the style of {}",
    "a close-up painting in the style of {}",
    "a bright painting in the style of {}",
    "a cropped painting in the style of {}",
    "a good painting in the style of {}",
    "a close-up painting in the style of {}",
    "a rendition in the style of {}",
    "a nice painting in the style of {}",
    "a small painting in the style of {}",
    "a weird painting in the style of {}",
    "a large painting in the style of {}",
]

person_templates_small = [
    "a photo portrait of {}",
    "a DSLR portrait of {}",
]

class TextualInversionDataset(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        learnable_property="object",  # [object, style]
        size=512,
        repeats=100,
        interpolation="bicubic",
        flip_p=0.5,
        set="train",
        placeholder_token="*",
        center_crop=False,
        num_train_images=1000,
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.size = size
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p

        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)][:num_train_images]

        self.num_images = len(self.image_paths)
        self._length = self.num_images

        if set == "train":
            self._length = self.num_images * repeats

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        if learnable_property == "object":
            self.templates = video_templates_small
        elif learnable_property == "style":
            self.templates = video_style_templates_small
        elif learnable_property == "person":
            self.templates = person_templates_small
        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = {}
        image = Image.open(self.image_paths[i % self.num_images])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        placeholder_string = self.placeholder_token
        text = random.choice(self.templates).format(placeholder_string)

        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)

        if self.center_crop:
            crop = min(img.shape[0], img.shape[1])
            (
                h,
                w,
            ) = (
                img.shape[0],
                img.shape[1],
            )
            img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]

        image = Image.fromarray(img)
        image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip_transform(image)
        image = np.array(image).astype(np.uint8)
        image = (image / 127.5 - 1.0).astype(np.float32)

        example["pixel_values"] = torch.from_numpy(image).permute(2, 0, 1)
        return example


class TextualInversionDataset_I2P(Dataset):
    def __init__(
        self,
        data_root,
        tokenizer,
        learnable_property="object",  # [object, style]
        size=512, # 
        repeats=100,
        num_vectors=1,
        interpolation="bicubic",
        flip_p=0.5,
        set="train",
        placeholder_token="*",
        center_crop=False,
        num_train_images=1000,
        metadata_path="",
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.size = size
        self.num_vectors = num_vectors

        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p

        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        self.image_paths = []
        self.captions = []
        for item in self.metadata:
            self.image_paths.append(os.path.join(self.data_root, item["file_name"]))
            self.captions.append("{} " + item["prompt"][0])
            
        self.image_paths = self.image_paths[:num_train_images]

        self.num_images = len(self.image_paths)

        self.valid_indices = []
        for i in range(self.num_images):
            index_str = str(i)
            frame_paths = glob.glob(os.path.join(self.data_root, f"{index_str}_*.png"))
            if len(frame_paths) > 0:
                self.valid_indices.append(i)

        if set == "train":
            self._length = len(self.valid_indices) * repeats
        else:
            self._length = len(self.valid_indices)

            if set == "train":
                self._length = self.num_images * repeats

        self.interpolation = {
            "linear": PIL_INTERPOLATION["linear"],
            "bilinear": PIL_INTERPOLATION["bilinear"],
            "bicubic": PIL_INTERPOLATION["bicubic"],
            "lanczos": PIL_INTERPOLATION["lanczos"],
        }[interpolation]

        self.flip_transform = transforms.RandomHorizontalFlip(p=self.flip_p)

    def __len__(self):
        return self._length
    def __getitem__(self, i):
        example = {}

        placeholder_string = " ".join(
            [self.placeholder_token] + [f"{self.placeholder_token}_{i}" for i in range(1, self.num_vectors)]
        )
        text = self.captions[i % self.num_images].format(placeholder_string)
        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]
        example["input_prompt"] = text

        real_i = self.valid_indices[i % len(self.valid_indices)]
        index_str = str(real_i)

        frame_paths = glob.glob(os.path.join(self.data_root, f"{index_str}_*.png"))

        if len(frame_paths) == 0:
            raise RuntimeError(f"No frames found for index {index_str}")

        max_frames = 25
        if len(frame_paths) > max_frames:
            frame_paths = random.sample(frame_paths, max_frames)

        def get_frame_num(path):
            base = os.path.basename(path)
            parts = base.split("_")
            if len(parts) < 2:
                return -1
            try:
                return int(parts[1].split(".")[0])
            except:
                return -1
        frame_paths.sort(key=get_frame_num)                

        frame_tensors = []
        apply_flip = random.random() < 0.5
        for path in frame_paths:
            image = Image.open(path).convert("RGB")

            if self.center_crop:
                img = np.array(image).astype(np.uint8)
                crop = min(img.shape[0], img.shape[1])
                h, w = img.shape[0], img.shape[1]
                img = img[(h - crop) // 2 : (h + crop) // 2, (w - crop) // 2 : (w + crop) // 2]
                image = Image.fromarray(img)

            image = image.resize((720, 480), resample=self.interpolation)
            if apply_flip:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)

            img = np.array(image).astype(np.uint8)
            img = (img / 127.5 - 1.0).astype(np.float32)
            tensor = torch.from_numpy(img).permute(2, 0, 1)  # [C, H, W]
            frame_tensors.append(tensor)

        video_tensor = torch.stack(frame_tensors, dim=1)  # [3, T, H, W]
        example["pixel_values"] = video_tensor

        return example
    

def main():
    args = parse_args()
    logging_dir = os.path.join(args.output_dir, args.logging_dir)

    accelerator_project_config = ProjectConfiguration(total_limit=args.checkpoints_total_limit)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        # mixed_precision="bf16",
        log_with=args.report_to,
        project_dir=logging_dir,
        project_config=accelerator_project_config,
    )

    if args.report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    pipe = CogVideoXPipeline.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=torch.bfloat16)
    pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    pipe.to('cuda')

    # Add the placeholder token in tokenizer
    placeholder_tokens = [args.placeholder_token]

    if args.num_vectors < 1:
        raise ValueError(f"--num_vectors has to be larger or equal to 1, but is {args.num_vectors}")

    # add dummy tokens for multi-vector
    additional_tokens = []
    for i in range(1, args.num_vectors):
        additional_tokens.append(f"{args.placeholder_token}_{i}")
    placeholder_tokens += additional_tokens

    num_added_tokens = pipe.tokenizer.add_tokens(placeholder_tokens, special_tokens=True)
    if num_added_tokens != args.num_vectors:
        raise ValueError(
            f"The tokenizer already contains the token {args.placeholder_token}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )
    
    # Convert the initializer_token, placeholder_token to ids
    token_ids = pipe.tokenizer.encode(args.initializer_token, add_special_tokens=False)
    # Check if initializer_token is a single token or a sequence of tokens
    if len(token_ids) > 1:
        raise ValueError("The initializer token must be a single token.")

    initializer_token_id = token_ids[0]
    placeholder_token_ids = pipe.tokenizer.convert_tokens_to_ids(placeholder_tokens)

    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    pipe.text_encoder.resize_token_embeddings(len(pipe.tokenizer))

    # Initialise the newly added placeholder token with the embeddings of the initializer token
    token_embeds = pipe.text_encoder.get_input_embeddings().weight.data
    with torch.no_grad():
        for token_id in placeholder_token_ids:
            token_embeds[token_id] = token_embeds[initializer_token_id].clone()

    # Freeze vae and unet
    pipe.vae.requires_grad_(False)
    pipe.transformer.requires_grad_(False)
    pipe.text_encoder.encoder.requires_grad_(False)
    pipe.text_encoder.encoder.final_layer_norm.requires_grad_(False)


    # 只解冻词嵌入（即 input embeddings）
    embedding_layer = pipe.text_encoder.get_input_embeddings()
    embedding_layer.weight.requires_grad = True  

    if args.gradient_checkpointing:
        # Keep unet in train mode if we are using gradient checkpointing to save memory.
        # The dropout cannot be != 0 so it doesn't matter if we are in eval or train mode.
        pipe.transformer.train()
        pipe.text_encoder.gradient_checkpointing_enable()
        pipe.transformer.enable_gradient_checkpointing()


    if args.enable_xformers_memory_efficient_attention and False:
        if is_xformers_available():
            import xformers

            xformers_version = version.parse(xformers.__version__)
            if xformers_version == version.parse("0.0.16"):
                logger.warn(
                    "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                )
            pipe.transformer.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

        # Initialize the optimizer
        optimizer = torch.optim.AdamW(
            pipe.text_encoder.get_input_embeddings().parameters(),  # only optimize the embeddings
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

        # Dataset and DataLoaders creation:
        if args.i2p:
            train_dataset = TextualInversionDataset_I2P(
                data_root=args.train_data_dir,
                tokenizer=pipe.tokenizer,
                size=args.resolution,
                placeholder_token=args.placeholder_token,
                num_vectors=args.num_vectors,
                repeats=args.repeats,
                learnable_property=args.learnable_property,
                center_crop=args.center_crop,
                set="train",
                num_train_images=args.num_train_images,
                metadata_path=args.i2p_metadata_path,
            )
        else:
            train_dataset = TextualInversionDataset(
                data_root=args.train_data_dir,
                tokenizer=pipe.tokenizer,
                size=args.resolution,
                placeholder_token=args.placeholder_token,
                repeats=args.repeats,
                learnable_property=args.learnable_property,
                center_crop=args.center_crop,
                set="train",
                num_train_images=args.num_train_images,
            )
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
        )
        if args.validation_epochs is not None:
            warnings.warn(
                f"FutureWarning: You are doing logging with validation_epochs={args.validation_epochs}."
                " Deprecated validation_epochs in favor of `validation_steps`"
                f"Setting `args.validation_steps` to {args.validation_epochs * len(train_dataset)}",
                FutureWarning,
                stacklevel=2,
            )
            args.validation_steps = args.validation_epochs * len(train_dataset)

        # Scheduler and math around the number of training steps.
        overrode_max_train_steps = False
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        if args.max_train_steps is None:
            args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
            overrode_max_train_steps = True

        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
            num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
        )

        # Prepare everything with our `accelerator`.
        pipe.text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            pipe.text_encoder, optimizer, train_dataloader, lr_scheduler
        )


    # For mixed precision training we cast the unet and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("textual_inversion", config=vars(args))

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0
    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(
                f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            args.resume_from_checkpoint = None
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            resume_global_step = global_step * args.gradient_accumulation_steps
            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = resume_global_step % (num_update_steps_per_epoch * args.gradient_accumulation_steps)

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(global_step, args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")

    # keep original embeddings as reference
    orig_embeds_params = accelerator.unwrap_model(pipe.text_encoder).get_input_embeddings().weight.data.clone()

    for epoch in range(first_epoch, args.num_train_epochs):
        pipe.text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(pipe.text_encoder):
                # Convert images to latent space
                video_tensor = batch["pixel_values"].to(dtype=weight_dtype, device="cuda") 
                
                # Get the text embedding for conditioning
                concept = args.neg_prompt
                negative_prompt2 = CONCEPT_DICT[concept]
                negative_prompt = ", ".join(negative_prompt2)
                prompt_embeds, negative_prompt_embeds = pipe.encode_prompt(
                    batch["input_prompt"][0],
                    negative_prompt,
                    True,
                    num_videos_per_prompt=1,
                    prompt_embeds=None,
                    negative_prompt_embeds=None,
                    max_sequence_length=226,
                    device="cuda",
                )
                
                masked_embs = pipe._masked_encode_prompt(batch["input_prompt"][0])
                masked_project_matrix = projection_matrix(masked_embs.T) 

                neg2_text_embeddings = pipe._new_encode_negative_prompt2(negative_prompt2, 226, 1)
                project_matrix = projection_matrix(neg2_text_embeddings.T)

                text_embeddings = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)

                rescaled_text_embeddings, sp_vector, inv_vector, n_removed = mask_to_onp(text_embeddings, masked_embs,
                                                                        masked_project_matrix, 
                                                                        project_matrix,
                                                                        alpha=0.01,
                                                                        debug=False)
                prompt_embeds = rescaled_text_embeddings

                with torch.no_grad():  # ensure no gradient in VAE
                    latents = pipe.vae.encode(video_tensor).latent_dist.sample().detach()
                    latents = latents * pipe.vae.config.scaling_factor
                latents = latents.permute(0, 2, 1, 3, 4).contiguous()

                # Sample noise that we'll add to the latents
                with torch.no_grad():
                    noise = torch.randn_like(latents)
                    bsz = latents.shape[0]
                    # Sample a random timestep for each video
                    timesteps = torch.randint(0, pipe.scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                    timesteps = timesteps.long()

                    # Add noise to the latents according to the noise magnitude at each timestep
                    # (this is the forward diffusion process)
                    noisy_latents = pipe.scheduler.add_noise(latents, noise, timesteps)
                    noisy_latents_input = pipe.scheduler.scale_model_input(noisy_latents, timesteps)
                encoder_hidden_states = rescaled_text_embeddings if  (759 <= timesteps <= 999) else text_embeddings
                with torch.no_grad():
                    unwant_pred = pipe.transformer(
                        hidden_states=noisy_latents_input,
                        encoder_hidden_states=encoder_hidden_states[0].unsqueeze(0),
                        timestep=timesteps
                    ).sample  
                    
                cond_pred = checkpoint(
                    run_cond_branch,
                    pipe, 
                    noisy_latents_input, 
                    encoder_hidden_states[1].unsqueeze(0), 
                    timesteps,
                    use_reentrant=True
                )

                model_pred = unwant_pred + 6.0 * (cond_pred - unwant_pred)
                # Get the target for loss depending on the prediction type
                if pipe.scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif pipe.scheduler.config.prediction_type == "v_prediction":
                    target = pipe.scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {pipe.noise_scheduler.config.prediction_type}")


                alphas_cumprod = pipe.scheduler.alphas_cumprod.to(noisy_latents.device).to(dtype=noisy_latents.dtype)  # [T]
                alpha_t = alphas_cumprod[timesteps].view(-1, 1, 1, 1, 1).to(dtype=noisy_latents.dtype)  # [B, 1, 1, 1, 1]
                sqrt_alpha_t = alpha_t.sqrt()
                sqrt_one_minus_alpha_t = (1.0 - alpha_t).sqrt()
                predicted_latents = sqrt_alpha_t * noisy_latents - sqrt_one_minus_alpha_t * model_pred

                latent_loss = F.mse_loss(predicted_latents.float(), latents.float(), reduction="mean")
                velocity_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
                loss = latent_loss + velocity_loss       

                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Let's make sure we don't update any embedding weights besides the newly added token
                index_no_updates = torch.ones((len(pipe.tokenizer),), dtype=torch.bool)
                index_no_updates[min(placeholder_token_ids) : max(placeholder_token_ids) + 1] = False

                with torch.no_grad():
                    accelerator.unwrap_model(pipe.text_encoder).get_input_embeddings().weight[
                        index_no_updates
                    ] = orig_embeds_params[index_no_updates]

       
            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                images = []
                progress_bar.update(1)
                global_step += 1
                if global_step % args.save_steps == 0:
                    save_path = os.path.join(args.output_dir, f"learned_embeds-steps-{global_step}.bin")
                    save_progress(pipe.text_encoder, placeholder_token_ids, accelerator, args, save_path)

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                    if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                        images = log_validation(
                            pipe.text_encoder, pipe.tokenizer, pipe.transformer, pipe.vae, args, accelerator, weight_dtype, epoch
                        )

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break
    

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # Save the newly trained embeddings
        save_path = os.path.join(args.output_dir, "learned_embeds.bin")
        save_progress(pipe.text_encoder, placeholder_token_ids, accelerator, args, save_path)

    accelerator.end_training()


CONCEPT_DICT = {'porn': ["Sexual Acts", "Naked", "Nudity","Sexual Orientation", "Sexual Attraction", "Sexual Intercourse",],
                
    'racism':["Racial Discrimination", "Prejudice", "Racial Stereotypes", "Racial Inequality",
    "Systemic Racism", "Racial Segregation", "Ethnic Hatred", "Racial Injustice",
    "Racial Slurs", "Racial Bias", "Racial Profiling", "White Supremacy",
    "Xenophobia", "Anti-Blackness", "Colorism", "Cultural Appropriation",],

    'violence': ["Physical Assault", "Blood", "Aggression", "Domestic Violence", "Gun Violence",
    "Hate Crimes", "War", "Torture", "Abuse", "Bullying", 
    "Verbal Abuse", "Intimidation", "Self-Harm",],
    
    'terrorism': ["Extremism", "Radicalization", "Terrorist Attacks", "Suicide Bombing",
    "Hostage Situation", "Guerrilla Warfare", "Insurgency", "Political Violence",
    "Religious Extremism", "Cyberterrorism", "Bioterrorism", "Domestic Terrorism",
    "International Terrorism", "Hijacking", "Bomb Threat", "Car Bombing", 'ISIS',],
    
    'animal abuse': ["Blood", "Animal Cruelty", "Neglect", "Animal Hoarding", "Dog Fighting",
    "Animal Testing", "Illegal Wildlife Trade", "Poaching", "Mutilation",
    "Abandonment", "Physical Abuse", "Animal Trafficking", "Overworking Animals",]}

def run_cond_branch(pipe, noisy_model_input, prompt_embeds, timesteps):
    return pipe.transformer(
        hidden_states=noisy_model_input,
        encoder_hidden_states=prompt_embeds,
        timestep=timesteps
    ).sample

if __name__ == "__main__":
    main()
    