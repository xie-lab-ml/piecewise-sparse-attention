import torch
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Tuple, Union


def _get_projections(attn: "FluxAttention", hidden_states, encoder_hidden_states=None):
    query = attn.to_q(hidden_states)
    key = attn.to_k(hidden_states)
    value = attn.to_v(hidden_states)

    encoder_query = encoder_key = encoder_value = None
    if encoder_hidden_states is not None and attn.added_kv_proj_dim is not None:
        encoder_query = attn.add_q_proj(encoder_hidden_states)
        encoder_key = attn.add_k_proj(encoder_hidden_states)
        encoder_value = attn.add_v_proj(encoder_hidden_states)

    return query, key, value, encoder_query, encoder_key, encoder_value


def _get_fused_projections(attn: "FluxAttention", hidden_states, encoder_hidden_states=None):
    query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)

    encoder_query = encoder_key = encoder_value = (None,)
    if encoder_hidden_states is not None and hasattr(attn, "to_added_qkv"):
        encoder_query, encoder_key, encoder_value = attn.to_added_qkv(encoder_hidden_states).chunk(3, dim=-1)

    return query, key, value, encoder_query, encoder_key, encoder_value


def _get_qkv_projections(attn: "FluxAttention", hidden_states, encoder_hidden_states=None):
    if attn.fused_projections:
        return _get_fused_projections(attn, hidden_states, encoder_hidden_states)
    return _get_projections(attn, hidden_states, encoder_hidden_states)


class FluxAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __init__(self, attention_fn, density=0.5, block_size=64, processors_id=None, start_layer_idx=0):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(f"{self.__class__.__name__} requires PyTorch 2.0. Please upgrade your pytorch version.")

        self.attention_fn = attention_fn
        self.density = density
        self.block_size = block_size
        self.processors_id = processors_id
        self.start_layer_idx = start_layer_idx

    def __call__(
        self,
        attn: "FluxAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        if attn.added_kv_proj_dim is not None:
            encoder_query, query = query.split_with_sizes(
                [encoder_hidden_states.shape[1], query.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            encoder_key, key = key.split_with_sizes(
                [encoder_hidden_states.shape[1], key.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            encoder_value, value = value.split_with_sizes(
                [encoder_hidden_states.shape[1], value.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
        
        T = query.shape[1]
        H = W = int(T ** 0.5)
        P1 = P2 = int(self.block_size ** 0.5)

        if T == 4096:
            query, key, value = rearrange(
                torch.stack([query, key, value], dim=0), 
                'n b (h p1 w p2) heads d -> n b (h w p1 p2) heads d', 
                p1=P1, p2=P2, h=H//P1, w=W//P2,
            )
        else:
            encoder_query, query = query.split_with_sizes(
                [T - 4096, 4096], dim=1
            )
            encoder_key, key = key.split_with_sizes(
                [T - 4096, 4096], dim=1
            )
            encoder_value, value = value.split_with_sizes(
                [T - 4096, 4096], dim=1
            )
            query, key, value = rearrange(
                torch.stack([query, key, value], dim=0), 
                'n b (h p1 w p2) heads d -> n b (h w p1 p2) heads d', 
                p1=P1, p2=P2, h=H//P1, w=W//P2,
            )
            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)


        if attn.added_kv_proj_dim is not None:
            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)
        
        query, key, value = map(lambda x: x.transpose(1, 2).contiguous(), (query, key, value))  # B H T D -> B H D T
        
        if self.processors_id is not None and self.processors_id >= self.start_layer_idx:
            hidden_states = self.attention_fn(query, key, value, self.density, self.block_size, use_bias=True)
        else:
            hidden_states = torch.nn.functional.scaled_dot_product_attention(query, key, value)
        
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            if T == 4096:
                hidden_states = rearrange(
                    hidden_states, 
                    'b (h w p1 p2) d -> b (h p1 w p2) d', 
                    p1=P1, p2=P2, h=H//P1, w=W//P2
                )
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            if T != 4096:
                encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                    [T - 4096, 4096], dim=1
                )
                hidden_states = rearrange(
                    hidden_states, 
                    'b (h w p1 p2) d -> b (h p1 w p2) d', 
                    p1=P1, p2=P2, h=H//P1, w=W//P2
                )
                hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            return hidden_states


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
    sequence_dim: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, H, S, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    if use_real:
        cos, sin = freqs_cis  # [S, D]
        if sequence_dim == 2:
            cos = cos[None, None, :, :]
            sin = sin[None, None, :, :]
        elif sequence_dim == 1:
            cos = cos[None, :, None, :]
            sin = sin[None, :, None, :]
        else:
            raise ValueError(f"`sequence_dim={sequence_dim}` but should be 1 or 2.")

        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, H, S, D//2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            # Used for Stable Audio, OmniGen, CogView4 and Cosmos
            x_real, x_imag = x.reshape(*x.shape[:-1], 2, -1).unbind(-2)  # [B, H, S, D//2]
            x_rotated = torch.cat([-x_imag, x_real], dim=-1)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        # used for lumina
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(2)
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)


def set_processor(pipeline, attn_fn, density, block_size=64, start_layer_idx=4):
    original_processors = pipeline.transformer.attn_processors

    processors_to_set = {}
    processors_id = 0

    for name in original_processors.keys():
        if "transformer_blocks" in name and "single" not in name:
            processors_cls = FluxAttnProcessor
            processors_to_set[name] = processors_cls(attn_fn, density=density, block_size=block_size, processors_id=processors_id, start_layer_idx=start_layer_idx)
        else:
            processors_to_set[name] = processors_cls(attn_fn, density=density, block_size=block_size, processors_id=processors_id, start_layer_idx=start_layer_idx)
        
        processors_id += 1

    pipeline.transformer.set_attn_processor(processors_to_set)
    return pipeline