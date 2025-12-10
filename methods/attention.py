
import torch
import torch.nn as nn
from typing import Optional
from transformers.processing_utils import Unpack
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers import BltModel
import math
class BLT_Transformer_Layer_Wrapper(nn.Module):
    def __init__(self, original_layer):
        super().__init__()
        self.original_layer = original_layer
        self.average_attention = None
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        cross_attention_states: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        full_text_row_masked_out_mask: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
        
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:

        residual = hidden_states
        hidden_states = self.original_layer.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.original_layer.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs
            )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.original_layer.post_attention_layernorm(hidden_states)
        hidden_states = self.original_layer.mlp(hidden_states)
        hidden_states = residual + hidden_states

        # Average self_attn_weights over heads and queries (B, H, N, N) → (B, N)
        avg_att = self_attn_weights.mean(dim=(1, 2))

        # Store the compact result
        self.average_attention = avg_att

        # Immediately free the huge tensor
        del self_attn_weights

        return hidden_states

class Model_Wrapper(nn.Module):

    def __init__(self, model, alfa):
        super().__init__()

        # Store model and alfa  
        self.model = model
        self.alfa = alfa

        # Get the BltModel inside
        self.blt_model = self._get_blt_model(model)

        # Wrap layers
        for i in range(len(self.blt_model.global_transformer.layers)):
            self.blt_model.global_transformer.layers[i] = BLT_Transformer_Layer_Wrapper(self.blt_model.global_transformer.layers[i])

        # Set attention implementation
        self.model.model.set_attn_implementation("eager")

    def _get_blt_model(self, model):
        """Drill down to the BltModel, regardless of LoRA wrapping"""
        base = model
        while not isinstance(base, BltModel):
            if hasattr(base, "model"):
                base = base.model
            else:
                raise ValueError(f"Cannot find BltModel in {type(model)}")
        return base

    def batch_entropy(self, batch):
        # Compute entropy along token dimension (B x N -> B)
        batch_entropies = - (batch * batch.clamp(min=1e-12).log()).sum(dim=1)
        return batch_entropies

    def get_loss(self):
        # Stack attentions from all layers (L x B x N)
        all_layers_attentions = torch.stack(
            [layer.average_attention for layer in self.blt_model.global_transformer.layers],
            dim=0)

        # Average over layers (B x N)
        average_attention = all_layers_attentions.mean(dim=0)

        # Entropy per sample (B)
        batch_entropies = self.batch_entropy(average_attention)

        # Negative entropy (scalar)
        loss = -batch_entropies.mean()

        # Scale by alfa
        return self.alfa * loss

    # Standard forward 
    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_attentions=True)
