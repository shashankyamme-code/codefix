import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math


class KVCachedMultiHeadAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_cache_len: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
<<<<<<< HEAD
        assert d_model % num_heads == 0
=======
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if max_cache_len <= 0:
            raise ValueError("max_cache_len must be positive")
>>>>>>> master

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_cache_len = max_cache_len

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    # -------------------------------------------------------------
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        cache: Optional[Dict[str, torch.Tensor]] = None,
        use_causal_mask: bool = True
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

<<<<<<< HEAD
        batch_size, seq_len, _ = query.shape
=======
        validate_inputs(query, key, value)
        if query.shape[-1] != self.d_model or key.shape[-1] != self.d_model:
            raise ValueError(f"Input feature dimensions must equal d_model ({self.d_model})")
        if query.shape[1] == 0 or key.shape[1] == 0:
            raise ValueError("Query, key, and value sequences must be non-empty")

        batch_size, seq_len, _ = query.shape
        cache_len = 0

        if cache is not None:
            cached_k = cache.get("key")
            cached_v = cache.get("value")
            if (cached_k is None) != (cached_v is None):
                raise ValueError("Cache must contain both key and value, or neither")
            if cached_k is not None:
                if (
                    cached_k.dim() != 3
                    or cached_v.dim() != 3
                    or cached_k.shape[0] != batch_size
                    or cached_v.shape[0] != batch_size
                    or cached_k.shape[2] != self.d_model
                    or cached_v.shape[2] != self.d_model
                    or cached_k.shape[1] != cached_v.shape[1]
                ):
                    raise ValueError(
                        "Cached key and value must have matching shape "
                        "(batch, cache_len, d_model)"
                    )
                if cached_k.device != key.device or cached_v.device != value.device:
                    raise ValueError("Cached key/value and input tensors must be on the same device")
                cache_len = cached_k.shape[1]
        if use_causal_mask and query.shape[1] != key.shape[1]:
            raise ValueError("Causal self-attention requires query and key lengths to match")
>>>>>>> master

        Q = self.q_proj(query)
        K_new = self.k_proj(key)
        V_new = self.v_proj(value)

<<<<<<< HEAD
        # ------------------------------------------
        # Cache handling (correct)
        # ------------------------------------------
        if cache is not None and cache.get("key") is not None:
            cached_k = cache["key"]
            cached_v = cache["value"]
=======
        if cache is not None and cache.get("key") is not None:
>>>>>>> master
            K = torch.cat([cached_k, K_new], dim=1)
            V = torch.cat([cached_v, V_new], dim=1)
        else:
            K = K_new
            V = V_new

<<<<<<< HEAD
        cache_len = K.shape[1] - seq_len

        # ------------------------------------------
        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
=======
        if K.shape[1] > self.max_cache_len:
            raise ValueError(
                f"Cache length {K.shape[1]} exceeds maximum length {self.max_cache_len}"
            )

        Q = self._split_heads(Q)
        K_heads = self._split_heads(K)
        V_heads = self._split_heads(V)

        scores = torch.matmul(Q, K_heads.transpose(-2, -1)) / self.scale
>>>>>>> master

        if use_causal_mask:
            scores = self._apply_causal_mask(scores, seq_len, cache_len)

        attention_weights = F.softmax(scores, dim=-1)

        if self.training:
            attention_weights = self.dropout(attention_weights)

<<<<<<< HEAD
        out = torch.matmul(attention_weights, V)
=======
        out = torch.matmul(attention_weights, V_heads)
>>>>>>> master
        out = self._merge_heads(out)
        out = self.out_proj(out)

        new_cache = {
            "key": K.detach(),
            "value": V.detach()
        }

<<<<<<< HEAD
        if new_cache["key"].shape[2] >= self.max_cache_len:
            raise ValueError("Cache exceeded maximum length")

=======
>>>>>>> master
        return out, new_cache

    # -------------------------------------------------------------
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.shape
        x = x.view(batch, seq, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    # -------------------------------------------------------------
    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, heads, seq, dim = x.shape
        x = x.permute(0, 2, 1, 3).contiguous()
        return x.view(batch, seq, self.d_model)

    # -------------------------------------------------------------
    def _apply_causal_mask(
        self,
        scores: torch.Tensor,
        seq_len: int,
        cache_len: int
    ):
        batch, heads, q_len, total_len = scores.shape
        mask = torch.triu(torch.ones(q_len, total_len, device=scores.device), 1 + cache_len)
        mask = mask.bool()
        scores = scores.masked_fill(mask, float("-inf"))
        return scores

    # -------------------------------------------------------------
    def reset_cache(self) -> Dict[str, Optional[torch.Tensor]]:
        return {"key": None, "value": None}

    # -------------------------------------------------------------
    def get_cache_info(self, cache: Optional[Dict[str, torch.Tensor]]):
<<<<<<< HEAD
        if cache is None or cache["key"] is None:
=======
        if cache is None:
>>>>>>> master
            return {
                "cache_length": 0,
                "cache_size_mb": 0.0,
                "is_full": False
            }

<<<<<<< HEAD
        key_cache = cache["key"]
        seq = key_cache.shape[2]
        mem = key_cache.numel() * key_cache.element_size() * 2
=======
        key_cache = cache.get("key")
        value_cache = cache.get("value")
        if key_cache is None and value_cache is None:
            return {
                "cache_length": 0,
                "cache_size_mb": 0.0,
                "is_full": False
            }
        if key_cache is None or value_cache is None:
            raise ValueError("Cache must contain both key and value")
        if key_cache.dim() != 3 or value_cache.dim() != 3 or key_cache.shape != value_cache.shape:
            raise ValueError("Cached key and value must have matching (batch, seq_len, d_model) shapes")
        seq = key_cache.shape[1]
        mem = (
            key_cache.numel() * key_cache.element_size()
            + value_cache.numel() * value_cache.element_size()
        )
>>>>>>> master
        return {
            "cache_length": seq,
            "cache_size_mb": round(mem / (1024 * 1024), 2),
            "is_full": seq >= self.max_cache_len
        }


# -------------------------------------------------------------
def compute_position_ids(seq_len: int, cache_len: int) -> torch.Tensor:
<<<<<<< HEAD
=======
    if seq_len < 0 or cache_len < 0:
        raise ValueError("Sequence and cache lengths must be non-negative")
>>>>>>> master
    return torch.arange(cache_len, cache_len + seq_len)


# -------------------------------------------------------------
def validate_inputs(query, key, value):
    if query.dim() != 3 or key.dim() != 3 or value.dim() != 3:
        raise ValueError("Inputs must be 3D")
    if query.shape[0] != key.shape[0] or query.shape[0] != value.shape[0]:
        raise ValueError("Batch must match")
<<<<<<< HEAD
    if key.shape[1] != value.shape[1]:
        raise ValueError("Key/value length mismatch")
=======
    if key.shape != value.shape:
        raise ValueError("Key and value must have matching shapes")
    if query.shape[2] != key.shape[2]:
        raise ValueError("Query, key, and value feature dimensions must match")
>>>>>>> master


# -------------------------------------------------------------
def create_sample_input(batch, seq, d_model, seed=None):
<<<<<<< HEAD
    if seed: torch.manual_seed(seed)
=======
    if seed is not None:
        torch.manual_seed(seed)
>>>>>>> master
    return (
        torch.randn(batch, seq, d_model),
        torch.randn(batch, seq, d_model),
        torch.randn(batch, seq, d_model),
    )


# -------------------------------------------------------------
if __name__ == "__main__":
    print("KV-Cached MHA ready.")
