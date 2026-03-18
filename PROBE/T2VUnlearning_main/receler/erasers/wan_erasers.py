import torch
import torch.nn as nn
import os
import json

import sys
sys.path.append('../../Wan2.2')
import wan
from wan.modules.model import WanAttentionBlock
from diffusers.models.attention import Attention
from typing import Any, Dict, Optional, Tuple, Union

from .utils import AdapterEraser

def save_wan_eraser_from_model(folder_path, model):
    
    difs_eraser_ckpt = {}
    eraser_rank = None
    for name, module in model.named_modules():
        if isinstance(module, WanWithEraser):
            eraser_name = f'{name}.adapter'
            if eraser_rank is None:
                eraser_rank = module.adapter.down.weight.shape[0]
            difs_eraser_ckpt[eraser_name] = module.adapter.state_dict()

    # save eraser weights
    os.makedirs(folder_path, exist_ok=True)
    eraser_weight_path = os.path.join(folder_path, f"eraser_weights.pt")
    torch.save(difs_eraser_ckpt, eraser_weight_path)
    

    # save eraser config
    eraser_config = {
        'eraser_type': 'adapter',
        'eraser_rank': eraser_rank,
    }
    eraser_config_path = os.path.join(folder_path, "eraser_config.json")
    with open(eraser_config_path, 'w') as f:
        json.dump(eraser_config, f, indent=4)


def setup_wan_adapter_eraser(model, eraser_rank, device, dtype):
    def replace_transformer_block(model):
        for name, module in model.named_modules():
            if isinstance(module, WanAttentionBlock):
                print("changing: ",name)
                original_attention = module.cross_attn
                modified_attention = WanWithEraser(original_attention, eraser_rank).to(device = device, dtype = dtype)
                module.cross_attn = modified_attention

    replace_transformer_block(model)
    erasers = {}
    for name, module in model.named_modules():
        if isinstance(module, WanWithEraser):
            eraser_name = f'{name}.adapter'
            print(eraser_name)
            erasers[eraser_name] = module.adapter
    return erasers

def inject_eraser_from_dict(model, erasers, eraser_rank):
    for name, module in model.named_modules():
        if isinstance(module, WanAttentionBlock):
            #print("changing: ",name)
            original_attention = module.cross_attn   
            modified_attention = WanWithEraser(original_attention, eraser_rank)
            module.cross_attn = modified_attention
            eraser_name = f'{name}.attn1.adapter'   
            module.cross_attn.adapter.load_state_dict(erasers[eraser_name].state_dict())
            module.cross_attn.adapter.to(device = model.device, dtype = model.dtype)


def inject_eraser(model, eraser_ckpt, eraser_rank, eraser_type='adapter'):
    for name, module in model.named_modules():
        if isinstance(module, WanAttentionBlock):
            print("changing: ",name)
            original_attention = module.cross_attn
            modified_attention = WanWithEraser(original_attention, eraser_rank)
            module.cross_attn = modified_attention
            eraser_name = f'{name}.cross_attn.{eraser_type}'
            module.cross_attn.adapter.load_state_dict(eraser_ckpt[eraser_name])
            module.cross_attn.adapter.to(device = model.device, dtype = model.dtype)
            #setattr(module, name, block_w_adapter)
        

class WanWithEraser(nn.Module):
    def __init__(
        self,
        attn,
        eraser_rank
    ):
        super().__init__()
        self.attn = attn
        self.adapter = AdapterEraser(attn.v.weight.shape[-1], eraser_rank)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        context_lens: Optional[torch.Tensor] = None,
        **cross_attention_kwargs,
    ) -> torch.Tensor:
        
        hidden_states = self.attn(
            hidden_states,
            encoder_hidden_states,
            context_lens,
            **cross_attention_kwargs,
        )

        if self.adapter.use_eraser:
            hidden_states = hidden_states + self.adapter(hidden_states)

        return hidden_states
    
