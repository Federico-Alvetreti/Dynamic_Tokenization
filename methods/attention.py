
import torch
import torch.nn as nn
from typing import Optional
from transformers.processing_utils import Unpack
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs


class BLT_Tranformer_Layer_Wrapper(nn.Module):
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

        print("ciao")


        return hidden_states
    

class Model_Wrapper(nn.Module):

    def __init__(self, model, alfa):
        super().__init__()

        # Build model 
        self.model = self.build_model(model)
        self.alfa = alfa
        model.model.set_attn_implementation("eager")

    # Create custom model 
    def build_model(self,model):
        # Wrap each layer 
        for i in range(len(model.model.global_transformer.layers)) :
            model.model.global_transformer.layers[i] = BLT_Tranformer_Layer_Wrapper(model.model.global_transformer.layers[i])
        return model
    

    # Get entropy of a batch B x N -> B x 1
    def batch_entropy(self, batch):
        batch_entropies = - (batch * batch.clamp(min=1e-12).log()).sum(dim=1)
        return batch_entropies 


    # Return the additional loss 
    def get_loss(self):
        
        # Stack the attentions from all layers (L, B, N)
        all_layers_attentions = torch.stack([layer.average_attention for layer  in self.model.model.global_transformer.layers], dim=0) # 
        
        # Get the average layer attention (B, N)
        average_attention = all_layers_attentions.mean(dim=0) 

        # Get the entropy of each sample (B)
        batch_entropies = self.batch_entropy(average_attention)

        # Get the negative entropy (1)
        loss = -batch_entropies.mean()

        # Scale by alfa
        scaled_loss = self.alfa * loss

        return scaled_loss


    # Standard forward 
    def forward(self, input_ids, attention_mask, labels):
        return self.model.forward(input_ids = input_ids,
                                  attention_mask = attention_mask,
                                  labels = labels,
                                  output_attentions=True)