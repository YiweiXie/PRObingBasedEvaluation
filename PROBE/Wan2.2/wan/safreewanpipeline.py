# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .distributed.sequence_parallel import sp_attn_forward, sp_dit_forward
from .distributed.util import get_world_size
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae2_2 import Wan2_2_VAE
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from .utils.utils import best_output_size, masks_like


def projection_matrix(E):
    """Calculate the projection matrix onto the subspace spanned by E."""   
    # P = E @ torch.pinverse(E.T @ E) @ E.T
    # P = E.bfloat16() @ torch.pinverse((E.T.bfloat16() @ E.bfloat16()).bfloat16()) @ E.T.bfloat16()
    # Convert to float for pinverse, then back to BFloat16 if needed
    # P = E.bfloat16() @ torch.pinverse((E.T.float() @ E.float()).float()) @ E.T.bfloat16()

    P = E.float() @ torch.pinverse((E.T.float() @ E.float()).float()) @ E.T.float()

    # P = E @ torch.pinverse((E.T @ E).float()) @ E.T
    return P

# 修改后的mask_to_onp_separate函数
def mask_to_onp_separate(context_null, context, p_emb, masked_input_subspace_projection, 
                        concept_subspace_projection, alpha=0., debug=False):
    """
    分别处理context_null和context
    """
    ms = masked_input_subspace_projection
    cs = concept_subspace_projection
    device = context_null[0].device
    (n_t, dim) = p_emb.shape   

    # SAFREE核心逻辑
    I_m_cs = torch.eye(dim).to(device) - cs
    dist_vec = I_m_cs.bfloat16() @ p_emb.T.bfloat16()
    dist_p_emb = torch.norm(dist_vec, dim=0)
    
    means = []
    for i in range(n_t):
        mean_without_i = torch.mean(torch.cat((dist_p_emb[:i], dist_p_emb[i+1:])))
        means.append(mean_without_i)

    mean_dist = torch.tensor(means).to(device)
    rm_vector = (dist_p_emb < (1. + alpha) * mean_dist).float()
    inv_vector = (dist_p_emb >= (1. + alpha) * mean_dist).float()
    
    n_removed = n_t - rm_vector.sum()
    print(f"Among {n_t} tokens, we remove {int(n_removed)}.")
    
    # 只对context应用SAFREE变换（context_null保持不变）
    text_seq_len = context[0].shape[0]
    ones_tensor_text = torch.ones(text_seq_len).to(device)
    if n_t + 1 <= text_seq_len:
        ones_tensor_text[1:n_t+1] = rm_vector
    ones_tensor_text = ones_tensor_text.unsqueeze(1)
    
    # SAFREE变换
    new_context = I_m_cs.bfloat16() @ ms.bfloat16() @ context[0].T.bfloat16()
    new_context = new_context.T
    merged_context = torch.where(ones_tensor_text.bool(), context[0], new_context)
    
    # context_null保持不变，只创建对应的mask
    ones_tensor_uncond = torch.ones(context_null[0].shape[0], 1).to(device)
    
    masks = (ones_tensor_uncond, ones_tensor_text)
    
    return context_null[0], merged_context, masks, n_removed.item()


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
    "Abandonment", "Physical Abuse", "Animal Trafficking", "Overworking Animals",], 
    
    'cassette_player': ["Cassette Player", "Tape Recorder", "Walkman", 
    "Portable Cassette", "Audio Cassette", 
    "Tape Deck", "Boom Box"], 

    'chain_saw': ["Chain Saw", "Chainsaw", "Power Saw", 
    "Gas Chainsaw", "Electric Chainsaw", "Portable Chainsaw",
    "Lumberjack Saw", "Tree Cutting Saw", "Wood Saw",
    "Stihl Chainsaw", "Husqvarna Chainsaw"],

    'church': ["Church", "Chapel", "Cathedral", "Basilica", 
    "Monastery", "Abbey", "Parish", "Sanctuary", 
    "House of Worship", "Christian Temple", "Orthodox Church",
    "Catholic Church", "Protestant Church"],

    'english_springer': [
    "English Springer", "English Springer Spaniel", 
    "Springer Spaniel", "Spaniel", 
    "Gun Dog", "Hunting Dog", "Sporting Dog",
    "Field Springer", "Show Springer"],

    'french_horn': [
    "French Horn", "Horn", "Double Horn", "Single Horn",
    "Vienna Horn", "Brass Horn", "Natural Horn", 
    "Orchestral Horn", "Mellophone"],

    'garbage_truck': [
    "Garbage Truck", "Trash Truck", "Waste Truck",
    "Rubbish Truck", "Refuse Truck", "Sanitation Truck",
    "Dump Truck", "Waste Collection Vehicle", "Refuse Collection Vehicle",
    "Recycling Truck", "Compactor Truck"],

    'gas_pump': [
    "Gas Pump", "Fuel Pump", "Petrol Pump", "Gasoline Pump",
    "Service Station Pump", "Fuel Dispenser", "Gas Station Pump",
    "Diesel Pump", "Petrol Station Pump"],

    'golf_ball': [
    "Golf Ball", "Ball for Golf", "Golfing Ball",
    "Practice Golf Ball", "Driving Range Ball", "Range Ball",
    "Tournament Golf Ball", "Dimpled Ball", "Titleist Golf Ball"],

    'parachute': [
    "Parachute", "Jumping Parachute", "Skydiving Parachute",
    "Military Parachute", "Rescue Parachute", "Sport Parachute",
    "Tandem Parachute", "Cargo Parachute", "Reserve Parachute",
    "Canopy"],

    'tench': [
    "Tench", "Tinca tinca", "Freshwater Tench", 
    "European Tench", "Tench Fish", "Common Tench",
    "Lake Tench", "Pond Tench"],

    'merkel': [
        "Angela Merkel", "Merkel", "Chancellor Merkel", "Frau Merkel",
        "Angela Dorothea Merkel", "German Chancellor", "CDU Leader",
        "Former German Chancellor"
    ],
    
    'obama': [
        "Barack Obama", "Obama", "President Obama", "Barack Hussein Obama II",
        "44th President", "Former US President", "Democratic President",
        "Obama Administration"
    ],
    
    'trump': [
        "Donald Trump", "Trump", "President Trump", "Donald J. Trump",
        "45th President", "Former President Trump", "Republican President",
        "Trump Administration"
    ],
    
    'biden': [
        "Joe Biden", "Biden", "President Biden", "Joseph Robinette Biden Jr.",
        "46th President", "Current US President", "Democratic President Biden",
        "Biden Administration"
    ],
    
    'queen': [
        "Queen Elizabeth II", "Queen Elizabeth", "Elizabeth II", "The Queen",
        "Her Majesty Queen Elizabeth II", "British Queen", "Monarch Elizabeth",
        "Queen of the United Kingdom"
    ],

    ### artist style
    "picasso": [
        "Picasso",
        "Cubism",
        "Cubist",
        "Guernica",
        "Les Demoiselles d'Avignon"
    ],
    
    "van_gogh": [
        "Van Gogh",
        "Post-impressionism",
        "Starry Night",
        "Sunflowers",
        "Impasto"
    ],
    
    "rembrandt": [
        "Rembrandt",
        "Baroque",
        "The Night Watch",
        "Chiaroscuro",
        "Dutch Golden Age"
    ],
    
    "andy_warhol": [
        "Warhol",
        "Pop art",
        "Campbell's Soup",
        "Marilyn Monroe",
        "Silkscreen"
    ],
    
    "caravaggio": [
        "Caravaggio",
        "Tenebrism",
        "The Calling of St Matthew",
        "Chiaroscuro",
        "Baroque realism"
    ]
    }



class WanTI2V_SAFREE:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_sp=False,
        t5_cpu=False,
        init_on_cpu=True,
        convert_model_dtype=False,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_sp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of sequence parallel.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
            init_on_cpu (`bool`, *optional*, defaults to True):
                Enable initializing Transformer Model on CPU. Only works without FSDP or USP.
            convert_model_dtype (`bool`, *optional*, defaults to False):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        if t5_fsdp or dit_fsdp or use_sp:
            self.init_on_cpu = False

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None)

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = Wan2_2_VAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device)

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model = WanModel.from_pretrained(checkpoint_dir)
        self.model = self._configure_model(
            model=self.model,
            use_sp=use_sp,
            dit_fsdp=dit_fsdp,
            shard_fn=shard_fn,
            convert_model_dtype=convert_model_dtype)

        if use_sp:
            self.sp_size = get_world_size()
        else:
            self.sp_size = 1

        self.sample_neg_prompt = config.sample_neg_prompt

    ### 这里是两个用于SAFREE的函数
    def _new_encode_negative_prompt2(self, negative_prompt2, max_length, num_images_per_prompt, pooler_output=True):
        device = "cuda"

        uncond_input = self.text_encoder.tokenizer(
            negative_prompt2,
            # padding="max_length",
            max_length=226,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
            return_mask=True,
        )
        
        uncond_embeddings = self.text_encoder.model(
            uncond_input[0].to(device),
            uncond_input[1].to(device),
        )

        if not pooler_output:
            uncond_embeddings = uncond_embeddings[0]
            bs_embed, seq_len, _ = uncond_embeddings.shape
            uncond_embeddings = uncond_embeddings.repeat(1, num_images_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(bs_embed * num_images_per_prompt, seq_len, -1)
        else:
            # uncond_embeddings = uncond_embeddings.last_hidden_state.mean(dim=1)#.pooler_output
            uncond_embeddings = uncond_embeddings[:,0,:]#.pooler_output

        return uncond_embeddings

    def _masked_encode_prompt(self, prompt):
        device = "cuda"
        
        # untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids
        print('prompt', len(prompt.split()))


        untruncated_ids = self.text_encoder.tokenizer(
            prompt,
            # padding="max_length",
            padding="longest",
            max_length=226,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        print('untruncated_ids', untruncated_ids.shape)

        n_real_tokens = untruncated_ids.shape[1] -2

        if untruncated_ids.shape[1] > 226:
            untruncated_ids = untruncated_ids[:, :226]
            n_real_tokens = 226 -2
        masked_ids = untruncated_ids.repeat(n_real_tokens, 1)

        for i in range(n_real_tokens):
            masked_ids[i, i+1] = 0

        ### 这里不太一致，对于wan来说，和cogvideo不一样，现在改了一版，可以test一下
        masked_embeddings = self.text_encoder.model(
            masked_ids.to(device),
            # attention_mask=None,
        )

        print('masked_embeddings', masked_embeddings.shape)

        # masked_embeddings = masked_embeddings.mean(dim=1)
        masked_embeddings = masked_embeddings[:,0,:]

        # print('masked_embeddings', masked_embeddings.shape)

        return masked_embeddings #.pooler_output
    ###

    def _configure_model(self, model, use_sp, dit_fsdp, shard_fn,
                         convert_model_dtype):
        """
        Configures a model object. This includes setting evaluation modes,
        applying distributed parallel strategy, and handling device placement.

        Args:
            model (torch.nn.Module):
                The model instance to configure.
            use_sp (`bool`):
                Enable distribution strategy of sequence parallel.
            dit_fsdp (`bool`):
                Enable FSDP sharding for DiT model.
            shard_fn (callable):
                The function to apply FSDP sharding.
            convert_model_dtype (`bool`):
                Convert DiT model parameters dtype to 'config.param_dtype'.
                Only works without FSDP.

        Returns:
            torch.nn.Module:
                The configured model.
        """
        model.eval().requires_grad_(False)

        if use_sp:
            for block in model.blocks:
                block.self_attn.forward = types.MethodType(
                    sp_attn_forward, block.self_attn)
            model.forward = types.MethodType(sp_dit_forward, model)

        if dist.is_initialized():
            dist.barrier()

        if dit_fsdp:
            model = shard_fn(model)
        else:
            if convert_model_dtype:
                model.to(self.param_dtype)
            if not self.init_on_cpu:
                model.to(self.device)

        return model

    def generate(self,
                 input_prompt,
                 img=None,
                 size=(1280, 704),
                 max_area=704 * 1280,
                 frame_num=81,
                 shift=5.0,
                 sample_solver='unipc',
                 sampling_steps=50,
                 guide_scale=5.0,
                 n_prompt="",
                 seed=-1,
                 offload_model=True):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            size (`tuple[int]`, *optional*, defaults to (1280,704)):
                Controls video resolution, (width,height).
            max_area (`int`, *optional*, defaults to 704*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 50):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # i2v
        if img is not None:
            return self.i2v(
                input_prompt=input_prompt,
                img=img,
                max_area=max_area,
                frame_num=frame_num,
                shift=shift,
                sample_solver=sample_solver,
                sampling_steps=sampling_steps,
                guide_scale=guide_scale,
                n_prompt=n_prompt,
                seed=seed,
                offload_model=offload_model)
        # t2v
        return self.t2v(
            input_prompt=input_prompt,
            size=size,
            frame_num=frame_num,
            shift=shift,
            sample_solver=sample_solver,
            sampling_steps=sampling_steps,
            guide_scale=guide_scale,
            n_prompt=n_prompt,
            seed=seed,
            offload_model=offload_model)

    def t2v(self,
            input_prompt,
            size=(1280, 704),
            frame_num=121,
            shift=5.0,
            sample_solver='unipc',
            sampling_steps=50,
            guide_scale=5.0,
            n_prompt="",
            seed=-1,
            offload_model=True):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (`tuple[int]`, *optional*, defaults to (1280,704)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 121):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 50):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # preprocess
        F = frame_num
        target_shape = (self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
                        size[1] // self.vae_stride[1],
                        size[0] // self.vae_stride[2])

        seq_len = math.ceil((target_shape[2] * target_shape[3]) /
                            (self.patch_size[1] * self.patch_size[2]) *
                            target_shape[1] / self.sp_size) * self.sp_size

        # if n_prompt == "":
        #     n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)


        ### 这里是做了SAFREE
        do_classifier_free_guidance = guide_scale > 1.0
        # concept = 'porn'

        negative_prompt2 = CONCEPT_DICT[n_prompt]
        # negative_prompt2 = CONCEPT_DICT[concept]
        n_prompt = ", ".join(negative_prompt2)
        ### 

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        ### 这里通过SAFREE得到掩码
        masked_embs = self._masked_encode_prompt(input_prompt)
        # print('masked_embs', masked_embs.shape)
        masked_project_matrix = projection_matrix(masked_embs.T) 

        neg2_text_embeddings = self._new_encode_negative_prompt2(negative_prompt2, 226, 1)
        project_matrix = projection_matrix(neg2_text_embeddings.T)
        
        ### 分别处理两个嵌入，而不是cat之后再处理
        processed_null, processed_context, masks, n_removed = mask_to_onp_separate(
            context_null, context, masked_embs, masked_project_matrix, project_matrix, alpha=0.01,
            debug=False
        )

        # if do_classifier_free_guidance:
        #     # print('hahaha')
        #     prompt_embeds = rescaled_text_embeddings #torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
        ### 

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g)
        ]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
                no_sync(),
        ):

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latents = noise
            mask1, mask2 = masks_like(noise, zero=False)

            arg_c = {'context': context, 'seq_len': seq_len}
            arg_null = {'context': context_null, 'seq_len': seq_len}

            if offload_model or self.init_on_cpu:
                self.model.to(self.device)
                torch.cuda.empty_cache()

            for i, t in enumerate(tqdm(timesteps)):
                ### 这里应该是需要对arg_c 和arg_null进行修改
                context_temp_null = [processed_null] if  (0 <= i <= 12) else context_null
                context_temp = [processed_context] if  (0 <= i <= 12) else context
                arg_c = {'context': context_temp, 'seq_len': seq_len}
                arg_null = {'context': context_temp_null, 'seq_len': seq_len}

                latent_model_input = latents
                timestep = [t]

                timestep = torch.stack(timestep)

                temp_ts = (mask2[0][0][:, ::2, ::2] * timestep).flatten()
                temp_ts = torch.cat([
                    temp_ts,
                    temp_ts.new_ones(seq_len - temp_ts.size(0)) * timestep
                ])
                timestep = temp_ts.unsqueeze(0)

                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0]
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null)[0]

                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latents = [temp_x0.squeeze(0)]
            x0 = latents
            if offload_model:
                self.model.cpu()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None

    def i2v(self,
            input_prompt,
            img,
            max_area=704 * 1280,
            frame_num=121,
            shift=5.0,
            sample_solver='unipc',
            sampling_steps=40,
            guide_scale=5.0,
            n_prompt="",
            seed=-1,
            offload_model=True):
        r"""
        Generates video frames from input image and text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation.
            img (PIL.Image.Image):
                Input image tensor. Shape: [3, H, W]
            max_area (`int`, *optional*, defaults to 704*1280):
                Maximum pixel area for latent space calculation. Controls video resolution scaling
            frame_num (`int`, *optional*, defaults to 121):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
                [NOTE]: If you want to generate a 480p video, it is recommended to set the shift value to 3.0.
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity.
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (121)
                - H: Frame height (from max_area)
                - W: Frame width (from max_area)
        """
        # preprocess
        ih, iw = img.height, img.width
        dh, dw = self.patch_size[1] * self.vae_stride[1], self.patch_size[
            2] * self.vae_stride[2]
        ow, oh = best_output_size(iw, ih, dw, dh, max_area)

        scale = max(ow / iw, oh / ih)
        img = img.resize((round(iw * scale), round(ih * scale)), Image.LANCZOS)

        # center-crop
        x1 = (img.width - ow) // 2
        y1 = (img.height - oh) // 2
        img = img.crop((x1, y1, x1 + ow, y1 + oh))
        assert img.width == ow and img.height == oh

        # to tensor
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device).unsqueeze(1)

        F = frame_num
        seq_len = ((F - 1) // self.vae_stride[0] + 1) * (
            oh // self.vae_stride[1]) * (ow // self.vae_stride[2]) // (
                self.patch_size[1] * self.patch_size[2])
        seq_len = int(math.ceil(seq_len / self.sp_size)) * self.sp_size

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
            oh // self.vae_stride[1],
            ow // self.vae_stride[2],
            dtype=torch.float32,
            generator=seed_g,
            device=self.device)

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # preprocess
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        z = self.vae.encode([img])

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
                no_sync(),
        ):

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latent = noise
            mask1, mask2 = masks_like([noise], zero=True)
            latent = (1. - mask2[0]) * z[0] + mask2[0] * latent

            arg_c = {
                'context': [context[0]],
                'seq_len': seq_len,
            }

            arg_null = {
                'context': context_null,
                'seq_len': seq_len,
            }

            if offload_model or self.init_on_cpu:
                self.model.to(self.device)
                torch.cuda.empty_cache()

            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = [latent.to(self.device)]
                timestep = [t]

                timestep = torch.stack(timestep).to(self.device)

                temp_ts = (mask2[0][0][:, ::2, ::2] * timestep).flatten()
                temp_ts = torch.cat([
                    temp_ts,
                    temp_ts.new_ones(seq_len - temp_ts.size(0)) * timestep
                ])
                timestep = temp_ts.unsqueeze(0)

                noise_pred_cond = self.model(
                    latent_model_input, t=timestep, **arg_c)[0]
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null)[0]
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latent = temp_x0.squeeze(0)
                latent = (1. - mask2[0]) * z[0] + mask2[0] * latent

                x0 = [latent]
                del latent_model_input, timestep

            if offload_model:
                self.model.cpu()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latent, x0
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None
