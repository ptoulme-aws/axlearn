from absl import logging
import jax
import jax.numpy as jnp
import functools
from functools import partial
import jax.numpy as jnp
import neuronxcc.nki.language as nl
import numpy as np
from jax_neuronx import nki_call
from neuronxcc.nki.kernels.attention import flash_attn_bwd, flash_fwd
from neuronxcc.nki._private_kernels.attention import attention_isa_kernel_cache, backward_attention_isa_kernel
from jax import custom_vjp
from neuronxcc.starfish.penguin.targets.nki.private_api import vnc
import os

@partial(custom_vjp, nondiff_argnums=(3,4))
def flash_attention(query, key, value, causal, softmax_scale):
  out, _ = _mha_forward(query, key, value, causal, softmax_scale)
  return out
  
def _mha_forward(query, key, value, causal, softmax_scale):
  # Get the batch size, sequence lengths, number of heads, and hidden dimension
  batch_size, q_seq_len, num_heads, d_model = query.shape
  _, kv_seq_len, _, _ = key.shape
  
  # Transpose the query, key, and value tensors
  q = query.transpose(0, 2, 3, 1)  # [batch_size, num_heads, d_model, q_seq_len]
  # FIXME: We're merging the batch_size and num_heads into a single dimension, is this valid?
  q = q.reshape(batch_size * num_heads, d_model, q_seq_len)
  k = key.transpose(0, 2, 3, 1)  # [batch_size, num_heads, d_model, kv_seq_len]
  k = k.reshape(batch_size * num_heads, d_model, kv_seq_len)
  v = value.transpose(0, 2, 1, 3)  # [batch_size, num_heads, kv_seq_len, d_model]
  v = v.reshape(batch_size * num_heads, kv_seq_len, d_model)
  
  # Create the output buffer
  attn_output_shape = jax.ShapeDtypeStruct((batch_size * num_heads, q_seq_len, d_model), dtype=query.dtype)
  # The BIR kernels don't produce an LSE, the LSE is recomputed in the backward kernel.
  # lse_shape = jax.ShapeDtypeStruct((batch_size, num_heads, nl.tile_size.pmax, q_seq_len // nl.tile_size.pmax), dtype=jnp.float32)
  # [bs, 128, seq_q / 128]
  recip_shape = jax.ShapeDtypeStruct((batch_size, nl.tile_size.pmax, q_seq_len // nl.tile_size.pmax), dtype=jnp.float32)
  neg_max_shape = jax.ShapeDtypeStruct((batch_size, nl.tile_size.pmax, q_seq_len // nl.tile_size.pmax), dtype=jnp.float32)
  seed = jnp.array([1])
  # Call the NKI kernel using nki_call
  print(f"ATTENTION HAPPENING")
  kernel_name = "CausalAttentionMMSoftmaxMMWithoutSwap" if causal else "AttentionMMSoftmaxMMWithoutSwap"
  # jax will attempt to turn into a TensorRef any args passed in the following nki_call. To pass constants that should not be
  # differentiated, such as for example softmax scale, include it in a partial(), such as partial(attention_isa_kernel_cache, scale=softmax_scale)
  # In the kernel interface being called here, such as attention_isa_kernel_cache, it is necessary for the constants to be keyword arguments
  attn_output, neg_max, recip = nki_call(
      partial(attention_isa_kernel_cache, scale=softmax_scale, kernel_name=kernel_name),
      q, k, v,
      out_shape=(attn_output_shape, neg_max_shape, recip_shape),
  )
  attn_output = attn_output.reshape(batch_size, num_heads, q_seq_len, d_model)
  # Transpose the output back to the original shape
  attn_output = attn_output.transpose(0, 2, 1, 3)  # [batch_size, q_seq_len, num_heads, d_model]

  return attn_output, (neg_max, recip, attn_output, q, k, v)

def _mha_backward(causal, softmax_scale, res, d_attn_output):
  # From fwd bir attn, o has shape [batch_size, q_seq_len, num_heads, d_model]
  neg_max, recip, o, q, k, v = res
  batch_size, seq_len_q, num_heads, d_model = o.shape
  _, _, seq_len_kv = k.shape

  # We expect [batch_size, d_model, seq_len]
  # Transpose the input tensors
  o = o.transpose(0, 2, 3, 1).reshape(batch_size * num_heads, d_model, seq_len_q)
  do = d_attn_output.transpose(0, 2, 3, 1).reshape(batch_size * num_heads, d_model, seq_len_q)

  # Transpose v tensor, expected shape from bir bwd kernel: [bs, d_head, seq_k]
  v = jnp.transpose(v, axes=(0, 2, 1))
  # Create the output buffer shapes
  d_query_shape = jax.ShapeDtypeStruct(q.shape, q.dtype)
  d_key_shape = jax.ShapeDtypeStruct(k.shape, k.dtype)
  d_value_shape = jax.ShapeDtypeStruct(v.shape, v.dtype)
  seed = jnp.array([1])
  print("BACK IS HAPPENING")

  # Call the NKI kernel using nki_call
  d_query, d_key, d_value = nki_call(
      partial(backward_attention_isa_kernel, is_causal=causal, dropout_p=0.0, scale_val=softmax_scale),
      q, k, v, o, do, neg_max, recip,
      out_shape=[d_query_shape, d_key_shape, d_value_shape],
  )

  # Batch seq_len heads, head_dim
  # Transpose the gradients back to the original shape
  # nki flash_attn_bwd outputs shape (bs, nheads, head_size, seq)
  # bir attn_bwd outputs shape [bs * num_heads, d_model, seq]
  d_query = d_query.reshape(batch_size, num_heads, d_model, seq_len_q).transpose(0, 3, 1, 2)
  d_key = d_key.reshape(batch_size, num_heads, d_model, seq_len_kv).transpose(0, 3, 1, 2)
  d_value = d_value.reshape(batch_size, num_heads, d_model, seq_len_kv).transpose(0, 3, 1, 2)

  return d_query, d_key, d_value

flash_attention.defvjp(_mha_forward, _mha_backward)
