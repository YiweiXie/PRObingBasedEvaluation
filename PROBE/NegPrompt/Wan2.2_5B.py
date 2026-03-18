"""Code adapted from: https://github.com/huggingface/diffusers/tree/main/examples/textual_inversion"""

import argparse
import logging
import math
import os
import random
import warnings
from pathlib import Path
import json
import glob
import numpy as np
import PIL
import re
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from huggingface_hub import create_repo, upload_folder

# TODO: remove and import from diffusers.utils when the new version of diffusers is released
from packaging import version
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
import torch.distributed as dist
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available

import sys
sys.path.append('../Wan2.2')

import wan
from wan.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.distributed.util import init_distributed_group
from wan.utils.prompt_extend import DashScopePromptExpander, QwenPromptExpander
from wan.utils.utils import merge_video_audio, save_video, str2bool, best_output_size, masks_like
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler


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


def save_progress(text_encoder, placeholder_token_ids, accelerator, args, save_path):
    logger.info("Saving embeddings")
    learned_embeds = (
        text_encoder.model.token_embedding.weight[min(placeholder_token_ids) : max(placeholder_token_ids) + 1]
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
        num_vectors=1,
        interpolation="bicubic",
        flip_p=0.5,
        set="train",
        neg_text="",
        placeholder_token="*",
        center_crop=False,
        num_train_images=1000,
    ):
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.learnable_property = learnable_property
        self.size = size
        self.num_vectors = num_vectors
        self.placeholder_token = placeholder_token
        self.center_crop = center_crop
        self.flip_p = flip_p
        self.neg_text = neg_text

        self.image_paths = [os.path.join(self.data_root, file_path) for file_path in os.listdir(self.data_root)][:num_train_images]

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
        placeholder_string = " ".join(
            [self.placeholder_token] + [f"{self.placeholder_token}_{i}" for i in range(1, self.num_vectors)]
        )
        text = random.choice(self.templates).format(placeholder_string)

        example["input_ids"] = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt",
        ).input_ids[0]

        real_i = self.valid_indices[i % len(self.valid_indices)]
        index_str = str(real_i)

        frame_paths = glob.glob(os.path.join(self.data_root, f"{index_str}_*.png"))
        if len(frame_paths) == 0:
            raise RuntimeError(f"No frames found for index {index_str}")
        
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
        myset="train",
        neg_text="",
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
        self.neg_text = neg_text

        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        self.image_paths = []
        self.captions = ["" for _ in range(num_train_images)]
        seen_indices = set()
        for item in self.metadata:
            self.image_paths.append(os.path.join(self.data_root, item["file_name"]))
            filename = os.path.basename(item["file_name"])
            index_str = filename.split("_")[0]              
            index_str = int(index_str)
            if index_str not in seen_indices:
                seen_indices.add(index_str)
                self.captions[index_str] = "{} "

            
        self.image_paths = self.image_paths[:num_train_images]

        self.num_images = len(self.image_paths)

        self.valid_indices = []
        for i in range(self.num_images):
            index_str = str(i)
            frame_paths = glob.glob(os.path.join(self.data_root, f"{index_str}_*.png"))
            if len(frame_paths) > 0:
                self.valid_indices.append(i)

        if myset == "train":
            self._length = len(self.valid_indices) * repeats
        else:
            self._length = len(self.valid_indices)

            if myset == "train":
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

        real_i = self.valid_indices[i % len(self.valid_indices)]
        index_str = str(real_i)
        frame_paths = glob.glob(os.path.join(self.data_root, f"{index_str}_*.png"))
        
        placeholder_string = " ".join(
            [self.placeholder_token] + [f"{self.placeholder_token}_{i}" for i in range(1, self.num_vectors)]
        )

        
        text = self.captions[real_i].format(placeholder_string)
        example["text"] = text       

        if len(frame_paths) == 0:
            raise RuntimeError(f"No frames found for index {index_str}")
        
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

            image = image.resize((1280, 704), resample=self.interpolation)
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
        log_with=args.report_to,
        project_dir=logging_dir,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

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


    cfg = WAN_CONFIGS["ti2v-5B"]
    rank = int(os.getenv("RANK", 0))
    offload_model = False
    size = "1280*704"

    pipe = wan.WanTI2V(
            config=cfg,
            checkpoint_dir=args.pretrained_model_name_or_path,
            device_id="0",
            rank=rank,
            t5_fsdp=False,
            dit_fsdp=False,
            use_sp=False,
            t5_cpu=False,   # default-False
            convert_model_dtype=True,
        )
    pipe.text_encoder.model.to("cuda:0")

    # Add the placeholder token in tokenizer
    placeholder_tokens = [args.placeholder_token]

    if args.num_vectors < 1:
        raise ValueError(f"--num_vectors has to be larger or equal to 1, but is {args.num_vectors}")

    # add dummy tokens for multi-vector
    additional_tokens = []
    for i in range(1, args.num_vectors):
        additional_tokens.append(f"{args.placeholder_token}_{i}")
    placeholder_tokens += additional_tokens

    num_added_tokens = pipe.text_encoder.tokenizer.tokenizer.add_tokens(placeholder_tokens, special_tokens=True)
    if num_added_tokens != args.num_vectors:
        raise ValueError(
            f"The tokenizer already contains the token {args.placeholder_token}. Please pass a different"
            " `placeholder_token` that is not already in the tokenizer."
        )

    # Convert the initializer_token, placeholder_token to ids
    token_ids = pipe.text_encoder.tokenizer.tokenizer.encode(args.initializer_token, add_special_tokens=False)
    # Check if initializer_token is a single token or a sequence of tokens
    if len(token_ids) > 1:
        raise ValueError("The initializer token must be a single token.")

    initializer_token_id = token_ids[0]
    placeholder_token_ids = pipe.text_encoder.tokenizer.tokenizer.convert_tokens_to_ids(placeholder_tokens)

    # Resize the token embeddings as we are adding new special tokens to the tokenizer
    token_nums = len(pipe.text_encoder.tokenizer.tokenizer)
    old_weight = pipe.text_encoder.model.token_embedding.weight
    vocab_size, dim = old_weight.shape
    new_emb = nn.Embedding(vocab_size + num_added_tokens, dim, device=old_weight.device, dtype=old_weight.dtype)
    new_emb.weight.data[:vocab_size] = old_weight
    new_emb.weight.data[vocab_size:] = torch.randn(num_added_tokens, dim, dtype=old_weight.dtype) * 0.02
    pipe.text_encoder.model.token_embedding.weight = new_emb.weight 

    # Initialise the newly added placeholder token with the embeddings of the initializer token
    token_embeds = pipe.text_encoder.model.token_embedding.weight.data
    with torch.no_grad():
        for token_id in placeholder_token_ids:
            token_embeds[token_id] = token_embeds[initializer_token_id].clone()

    # Freeze vae and unet
    pipe.vae.model.requires_grad_(False)
    pipe.model.requires_grad_(False)
    pipe.text_encoder.model.requires_grad_(False)

    embedding_layer = pipe.text_encoder.model.token_embedding
    embedding_layer.weight.requires_grad = True 

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
            pipe.text_encoder.model.token_embedding.parameters(),  # only optimize the embeddings
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

        # Dataset and DataLoaders creation:
        if args.i2p:
            train_dataset = TextualInversionDataset_I2P(
                data_root=args.train_data_dir,
                tokenizer=pipe.text_encoder.tokenizer,
                size=args.resolution,
                placeholder_token=args.placeholder_token,
                num_vectors=args.num_vectors,
                repeats=args.repeats,
                neg_text=args.neg_prompt,
                learnable_property=args.learnable_property,
                center_crop=args.center_crop,
                myset="train",
                num_train_images=args.num_train_images,
                metadata_path=args.i2p_metadata_path,
            )
        else:
            train_dataset = TextualInversionDataset(
                data_root=args.train_data_dir,
                tokenizer=pipe.text_encoder.tokenizer,
                size=args.resolution,
                placeholder_token=args.placeholder_token,
                repeats=args.repeats,
                neg_text=args.neg_prompt,
                learnable_property=args.learnable_property,
                num_vectors=args.num_vectors,
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
        pipe.text_encoder.model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            pipe.text_encoder.model, optimizer, train_dataloader, lr_scheduler
        )


    # For mixed precision training we cast the unet and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move vae and unet to device and cast to weight_dtype
    pipe.model.to(accelerator.device, dtype=weight_dtype)

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
    orig_embeds_params = accelerator.unwrap_model(pipe.text_encoder.model).token_embedding.weight.data.clone()
    pipe.model.to("cuda:1")
    for epoch in range(first_epoch, args.num_train_epochs):
        pipe.model.eval()
        pipe.vae.model.eval()
        pipe.text_encoder.model.token_embedding.train()
        for step, batch in enumerate(train_dataloader):
            # Skip steps until we reach the resumed step
            if args.resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % args.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            with accelerator.accumulate(pipe.text_encoder):
                
                print(batch["text"])
                video_tensor = batch["pixel_values"].to(dtype=weight_dtype, device="cuda:0")
                video_list = [video_tensor.squeeze(0)]

                target_shape = [48, 7, 44, 80]
                seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (pipe.patch_size[1] * pipe.patch_size[2]) *
                            target_shape[1] / pipe.sp_size) * pipe.sp_size
                
                # Get the text embedding for conditioning
                encoder_hidden_states = pipe.text_encoder(batch["text"], "cuda:0")
                neg_prompt = pipe.text_encoder("nudity", "cuda:0")
                encoder_hidden_states = [encoder_hidden_states[0].to("cuda:1")]
                neg_prompt = [neg_prompt[0].to("cuda:1")]

                with torch.no_grad():  # ensure no gradient in VAE
                    latents = pipe.vae.encode(video_list)

                latents = [latents[0].to("cuda:1")]
                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents[0])
                bsz = 1

                sampling_steps = 1000
                shift = 5.0
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=pipe.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=latents[0].device, shift=shift)
                timesteps = sample_scheduler.timesteps
                sigmas = sample_scheduler.sigmas
                # Sample a random timestep for each video
                idx = torch.randint(0, sampling_steps, (bsz,), device=latents[0].device)
                t = timesteps[idx]

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = sample_scheduler.add_noise(latents[0], noise, t)
                noisy_latents = [noisy_latents]
                mask1, mask2 = masks_like([noise], zero=False)

                arg_c = {'context': encoder_hidden_states, 'seq_len': seq_len}
                arg_neg = {'context': neg_prompt, 'seq_len': seq_len}

                temp_ts = (mask2[0][0][:, ::2, ::2] * t).flatten()
                temp_ts = torch.cat([
                    temp_ts,
                    temp_ts.new_ones(seq_len - temp_ts.size(0)) * t
                ])
                timestep = temp_ts.unsqueeze(0)


                # Predict the noise residual
                with torch.amp.autocast('cuda:1', dtype=pipe.param_dtype):
                    with torch.no_grad():
                        # neg prompt
                        noise_pred_neg = pipe.model(
                            noisy_latents, t=timestep, **arg_neg)[0]

                    noise_pred_cond = pipe.model(
                            noisy_latents, t=timestep, **arg_c)[0]
                
                noise_pred = noise_pred_neg + 5.0 * (
                    noise_pred_cond - noise_pred_neg)
                
                sample_scheduler._init_step_index(t)
                pred_x0 = sample_scheduler.convert_model_output(
                noise_pred, sample=noisy_latents[0])

                # predicted_latents and target_latents：
                latent_loss = F.mse_loss(pred_x0.float(), latents[0].float(), reduction="mean")
                velocity_loss = F.mse_loss(noise_pred.float(), (noise - latents[0]).float(), reduction="mean")
                loss = velocity_loss + latent_loss  

                accelerator.backward(loss)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                # Let's make sure we don't update any embedding weights besides the newly added token
                index_no_updates = torch.ones((vocab_size + args.num_vectors,), dtype=torch.bool)
                index_no_updates[min(placeholder_token_ids) : max(placeholder_token_ids) + 1] = False

                with torch.no_grad():
                    accelerator.unwrap_model(pipe.text_encoder.model).token_embedding.weight[
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


if __name__ == "__main__":
    main()
    