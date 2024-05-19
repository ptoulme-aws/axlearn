import jax
import jax.numpy as jnp
import functools
from functools import partial
import jax.numpy as jnp
import neuronxcc.nki.language as nl
import numpy as np
from neuron_jax import nki_call


"""
Copyright (c) 2023, Amazon.com. All Rights Reserved

kernels - Builtin high performance attention kernels

"""
import numpy as np

from neuronxcc.nki import trace
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.nccl as nccl
import neuronxcc.nki.language as nl

from neuronxcc.nki.language import par_dim
from neuronxcc.starfish.penguin.common import div_ceil
from dataclasses import dataclass

def fused_self_attn(q_ref, k_ref, v_ref, out_ref, use_causal_mask=False, mixed_percision=True):
  """
  Fused self attention kernel. Computes softmax(QK^T)V. Decoder model
  can optionally include a causal mask application. Does not include QKV
  projection, output projection, dropout, residual connection, etc.

  IO tensor layouts:
   - q_ptr: shape (bs, nheads, head_size, seq_q)
   - k_ptr: shape (bs, nheads, head_size, seq_k)
   - v_ptr: shape (bs, nheads, head_size, seq_v)
   - out_ptr: shape (bs, nheads, head_size, seq_q)
   - We use seq_q and seq_k just for clarity, this kernel requires seq_q == seq_k

  IO tensor dtypes:
   - This kernel assumes all IO tensors have the same dtype
   - Intermediate tensor dtypes will use the same dtype as IO tensors

  We can assume K/V/Q tensors (per attention head, per batch sample) have the same matmul
  output layout: [seqlen F, d_head P]

  The attention kernel has the following computation/data reshape:

  1. transpose(tensor_v) = trans_v[d_head F, seqlen_v P] (compiler maps this to dma_transpose)

   - We put this as first thing in the kernel because we want to cache the full tensor_v in SBUF
   - that can be reused across different softmax tiles in the same attention head/batch.
   - If we put it in the inner loop of attention, we will be reloading the same tensor_v seqlen/128 times.

  2. matmul_0(lhs=tensor_k, rhs=tensor_q, contract=d_head)
     = qk_result[seqlen_k F, seqlen_q_tile P]

   - Input layout consumed as is
   - Swapping lhs/rhs avoids an explicit transpose of Q@K_T
   - Process all of seqlen_k from tensor_k in inner loop, 128x128 tile for tensor_q

  3. causal_mask (optional, decoder model only)

   - affine_select on qk_result

  4. softmax

   - tensor_reduce_max(qk_result, negate=1) = max_qk_negated[1 F, seqlen_q_tile P]
   - act_exp(qk_result - max_qk_negated) = exp_res[seqlen_k F, seqlen_q_tile P]
   - tensor_reduce_add(exp_res) = sum_res[1F, seqlen_q_tile P] (compiler folds this with act_exp)
   - reciprocal(sum_res) = inverse_sum_res[1F, seqlen_q_tile P]
   - tensor_scalar_multiply(exp_res, inverse_sum_res) = softmax_res[seqlen_k F, seqlen_q_tile P]

  5. transpose(softmax_res) = trans_softmax_res[seqlen_q_tile F, seqlen_k P]
  6. matmul_1(lhs=trans_softmax_res, rhs=trans_v, contract=seqlen_v=seqlen_k)
     = attn_result[seqlen_q_tile F, d_head P]


  Some naming convention used:
   - ip_[name] and if_[name] are tile indices returned from nl.arange(partition dim size) and nl.arange(free dim size).
     ip_[name] and if_[name] represent the partition and free indices respectively. Such tile indices are typically used
     to slice a tensor for a Neuron instruction as input/output. We also use these tile indices to build affine expressions
     for IO tensor accesses and predicates/masks.
  """

  # Use q_ref dtype as the intermediate tensor dtype
  # Assume all IO tensors have the same dtype
  kernel_dtype = q_ref.dtype
  pe_in_dt = nl.bfloat16 if mixed_percision else np.float32
  assert q_ref.dtype == k_ref.dtype == v_ref.dtype == out_ref.dtype

  # Shape checking
  bs, nheads, d_head, seqlen = q_ref.shape
  assert tuple(k_ref.shape) == (bs, nheads, d_head, seqlen), 'Input shape mismatch!'
  assert tuple(v_ref.shape) == (bs, nheads, d_head, seqlen), 'Input shape mismatch!'
  assert tuple(q_ref.shape) == (bs, nheads, d_head, seqlen), 'Input shape mismatch!'
  assert tuple(out_ref.shape) == (bs, nheads, d_head, seqlen), 'Output shape mismatch!'

  # Softmax scaling factor, multiplied onto Q
  softmax_scale = 1.0 / float(d_head ** 0.5)

  # Different batch samples/attention heads have independent attention
  batch_id = nl.program_id(axis=0)
  head_id = nl.program_id(axis=1)

  # TODO: make q_seq_tile_size user input
  # The matmuls currently use a fixed tile size of (128, 128). This may not achieve the best
  # performance for dense attention. However, since this kernel is in preparation
  # for block-sparse attention, this tile size is acceptable because the block
  # size of block-sparse attention cannot be too large.
  q_seq_n_tiles, q_seq_tile_size = seqlen // 128, 128
  k_seq_n_tiles, k_seq_tile_size = seqlen // 128, 128
  d_head_n_tiles, d_head_tile_size = d_head // 128, 128
  v_seq_n_tiles, v_seq_tile_size = seqlen // 128, 128

  ###################################
  # Step 1. transpose(tensor_v)
  ###################################
  # Buffer for v matrix transposed
  # Pre-fetch and keep it in SBUF throughout different softmax tiles
  trans_v = nl.ndarray((par_dim(v_seq_tile_size), v_seq_n_tiles, d_head), dtype=pe_in_dt)

  for i_d_head_tile in nl.affine_range(d_head_n_tiles):
    for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
      ip_v = nl.arange(d_head_tile_size)[:, None]
      if_v = nl.arange(k_seq_tile_size)[None, :]
      ip_v_t = nl.arange(k_seq_tile_size)[:, None]
      if_v_t = nl.arange(d_head_tile_size)[None, :]
      v_local = nl.load(
        v_ref[
          batch_id, head_id, i_d_head_tile * d_head_tile_size + ip_v, i_k_seq_tile * k_seq_tile_size + if_v],
        dtype=pe_in_dt)
      trans_v[ip_v_t, i_k_seq_tile, i_d_head_tile * d_head_tile_size + if_v_t] = nisa.sb_transpose(
        v_local)

  q_local = nl.ndarray((q_seq_n_tiles, d_head_n_tiles, par_dim(d_head_tile_size), q_seq_tile_size),
                       dtype=pe_in_dt)
  ip_q = nl.arange(d_head_tile_size)[:, None]
  if_q = nl.arange(q_seq_tile_size)[None, :]
  for i_d_head_tile in nl.affine_range(d_head_n_tiles):
    for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):
      q_local[i_q_seq_tile, i_d_head_tile, ip_q, if_q] = nl.load(
        q_ref[
          batch_id, head_id, i_d_head_tile * d_head_tile_size + ip_q, i_q_seq_tile * q_seq_tile_size + if_q],
        dtype=pe_in_dt) * softmax_scale

  k_local = nl.ndarray((k_seq_n_tiles, d_head_n_tiles, par_dim(d_head_tile_size), k_seq_tile_size),
                       dtype=pe_in_dt)
  ip_k = nl.arange(d_head_tile_size)[:, None]
  if_k = nl.arange(k_seq_tile_size)[None, :]
  for i_d_head_tile in nl.affine_range(d_head_n_tiles):
    for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
      k_local[i_k_seq_tile, i_d_head_tile, ip_k, if_k] = nl.load(
        k_ref[
          batch_id, head_id, i_d_head_tile * d_head_tile_size + ip_k, i_k_seq_tile * k_seq_tile_size + if_k],
        dtype=pe_in_dt)

  for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):  # indent = 2
    # A SBUF buffer for an independent softmax tile
    qk_res_buf = nl.ndarray((par_dim(q_seq_tile_size), seqlen), dtype=kernel_dtype)

    neg_max_res = nl.ndarray((par_dim(q_seq_tile_size), k_seq_n_tiles), dtype=kernel_dtype)
    ip_max = nl.arange(q_seq_tile_size)[:, None]
    if_max = nl.arange(k_seq_n_tiles)[None, :]

    # Loop over LHS free of matmul(lhs=tensor_k, rhs=tensor_q, contract=d_head)
    for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):  # indent = 4

      # Since the K^T tile is the LHS, the q_seq_len dimension will be P in the result
      # PSUM buffer shape: [q_seq_tile_size P, k_seq_tile_size F]
      qk_psum = nl.zeros((par_dim(q_seq_tile_size), k_seq_tile_size),
                         dtype=np.float32, buffer=nl.psum)

      # Tensor indices for accessing qk result in k_seq_tile_size
      ip_qk = nl.arange(q_seq_tile_size)[:, None]
      if_qk = nl.arange(k_seq_tile_size)[None, :]

      # Loop over contraction dim of Step 1 matmul
      for i_d_head_tile in nl.affine_range(d_head_n_tiles):  # indent = 6
        ##############################################################
        # Step 2. matmul(lhs=tensor_k, rhs=tensor_q, contract=d_head)
        ##############################################################
        qk_psum[ip_qk, if_qk] += nisa.nc_matmul(rhs=k_local[i_k_seq_tile, i_d_head_tile, ip_k, if_k],
                                                lhs=q_local[i_q_seq_tile, i_d_head_tile, ip_q, if_q])

        ###################################
        # Step 3. Apply optional causal mask
        ###################################
        if use_causal_mask:
          # Magic number -9984.0 to replace -inf similar to what Tensorizer uses
          qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk] = nisa.affine_select(
            pred=(i_q_seq_tile * q_seq_tile_size + ip_qk >= i_k_seq_tile * k_seq_tile_size + if_qk),
            on_true_tile=qk_psum[ip_qk, if_qk], on_false_value=-9984.0, dtype=kernel_dtype)
        else:
          # Simply send psum result back to sbuf
          qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk] = nl.copy(
            qk_psum[ip_qk, if_qk],
            dtype=kernel_dtype)

      ###################################
      # Step 4. Softmax
      ###################################
      # TODO: use TensorScalarCacheReduce to avoid an extra copy
      # We want to break this reduction in tiles because we want to overlap it with the previous matmul
      neg_max_res[ip_max, i_k_seq_tile] = nisa.reduce(
        np.max, data=qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk],
        axis=(1,), dtype=kernel_dtype, negate=True)

    neg_max_res_final = nisa.reduce(
      np.min, data=neg_max_res[ip_max, if_max],
      axis=(1,), dtype=kernel_dtype, negate=False)

    ip_softmax = nl.arange(q_seq_tile_size)[:, None]
    if_softmax = nl.arange(seqlen)[None, :]
    ip_sum_res = nl.arange(q_seq_tile_size)[:, None]
    if_sum_res = nl.arange(d_head_tile_size)[None, :]

    softmax_res = nl.ndarray((par_dim(q_seq_tile_size), seqlen), dtype=pe_in_dt)
    sum_divisor = nl.ndarray((par_dim(q_seq_tile_size), d_head_tile_size), dtype=kernel_dtype)

    # Simply use a large tile of seq_len in size since this is a "blocking" instruction
    # Assuming the compiler will merge exp and reduce_add into a single instruction on ACT
    exp_res = nisa.activation(np.exp,
                              data=qk_res_buf[ip_softmax, if_softmax],
                              bias=neg_max_res_final, scale=1.0)

    sum_res = nisa.reduce(np.add, data=exp_res, axis=(1,),
                          dtype=kernel_dtype)
    softmax_res[ip_softmax, if_softmax] = nl.copy(exp_res, dtype=pe_in_dt)

    sum_reciprocal_broadcast = (1.0 / sum_res).broadcast_to((q_seq_tile_size, d_head_tile_size))
    sum_divisor[ip_sum_res, if_sum_res] = nl.copy(sum_reciprocal_broadcast, dtype=kernel_dtype)

    # Loop over matmul_1 RHS free
    for i_d_head_tile in nl.affine_range(d_head_n_tiles):

      # Buffer for transposed softmax results (FP32 in PSUM)
      trans_softmax_res = nl.ndarray(
        (par_dim(k_seq_tile_size), k_seq_n_tiles, q_seq_tile_size),
        dtype=pe_in_dt)

      # Result psum buffer has the hidden dim as P
      attn_res_psum = nl.zeros((par_dim(d_head_tile_size), q_seq_tile_size),
                               dtype=np.float32, buffer=nl.psum)

      ip_scores_t = nl.arange(k_seq_tile_size)[:, None]
      if_scores_t = nl.arange(q_seq_tile_size)[None, :]
      # Loop over matmul_1 contraction
      for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
        ###################################
        # Step 5. transpose(softmax_res)
        ###################################
        ip_scores = nl.arange(q_seq_tile_size)[:, None]
        if_scores = nl.arange(k_seq_tile_size)[None, :]

        trans_softmax_res[ip_scores_t, i_k_seq_tile, if_scores_t] = nisa.sb_transpose(
          softmax_res[ip_scores, i_k_seq_tile * k_seq_tile_size + if_scores])

      ip_out = nl.arange(d_head_tile_size)[:, None]
      if_out = nl.arange(q_seq_tile_size)[None, :]
      for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
        ######################################################################
        # Step 6. matmul_1(lhs=trans_softmax_res, rhs=trans_v, contract=seqlen_v=seqlen_k)
        ######################################################################
        ip_v_t = nl.arange(k_seq_tile_size)[:, None]
        if_v_t = nl.arange(d_head_tile_size)[None, :]
        attn_res_psum[ip_out, if_out] += \
          nisa.nc_matmul(rhs=trans_softmax_res[ip_scores_t, i_k_seq_tile, if_scores_t],
                         lhs=trans_v[ip_v_t, i_k_seq_tile, i_d_head_tile * d_head_tile_size + if_v_t])

      attn_res_sbuf = nl.copy(attn_res_psum[ip_out, if_out], dtype=kernel_dtype)

      attn_res_div = attn_res_sbuf * nisa.sb_transpose(sum_divisor[ip_sum_res, if_sum_res])
      nl.store(
        out_ref[
          batch_id, head_id, i_d_head_tile * d_head_tile_size + ip_out, i_q_seq_tile * q_seq_tile_size + if_out],
        value=attn_res_div)


def fused_self_attn_fwd_cache_softmax(
    q_ref, k_ref, v_ref,
    seed_ref,
    out_ref,
    out_cached_negative_max_ref,
    out_cached_sum_reciprocal_ref,
    use_causal_mask=True,
    mixed_precision=True,
    dropout_p=0.0):
  """
  Fused self attention kernel. Computes softmax(QK^T)V. Decoder model
  can optionally include a causal mask application. Does not include QKV
  projection, output projection, dropout, residual connection, etc.

  IO tensor layouts:
   - q_ref: shape (bs, nheads, head_size, seq_q)
   - k_ref: shape (bs, nheads, head_size, seq_k)
   - v_ref: shape (bs, nheads, head_size, seq_v)
   - seed_ref: shape (1, )
   - out_ref: shape (bs, nheads, head_size, seq_q)
   - out_cached_negative_max_ref: shape (bs, nheads, 128, seq_q // 128)
   - out_cached_sum_reciprocal_ref: shape (bs, nheads, 128, seq_q // 128)
   - We use seq_q and seq_k just for clarity, this kernel requires seq_q == seq_k

  IO tensor dtypes:
   - This kernel assumes all IO tensors have the same dtype
   - Intermediate tensor dtypes will use the same dtype as IO tensors

  We can assume K/V/Q tensors (per attention head, per batch sample) have the same matmul
  output layout: [seqlen F, d_head P]

  The attention kernel has the following computation/data reshape:

  1. transpose(tensor_v) = trans_v[d_head F, seqlen_v P] (compiler maps this to dma_transpose)

   - We put this as first thing in the kernel because we want to cache the full tensor_v in SBUF
   - that can be reused across different softmax tiles in the same attention head/batch.
   - If we put it in the inner loop of attention, we will be reloading the same tensor_v seqlen/128 times.

  2. matmul_0(lhs=tensor_k, rhs=tensor_q, contract=d_head)
     = qk_result[seqlen_k F, seqlen_q_tile P]

   - Input layout consumed as is
   - Swapping lhs/rhs avoids an explicit transpose of Q@K_T
   - Process all of seqlen_k from tensor_k in inner loop, 128x128 tile for tensor_q

  3. causal_mask (optional, decoder model only)

   - affine_select on qk_result

  4. softmax

   - tensor_reduce_max(qk_result, negate=1) = max_qk_negated[1 F, seqlen_q_tile P]
   - act_exp(qk_result - max_qk_negated) = exp_res[seqlen_k F, seqlen_q_tile P]
   - tensor_reduce_add(exp_res) = sum_res[1F, seqlen_q_tile P] (compiler folds this with act_exp)
   - reciprocal(sum_res) = inverse_sum_res[1F, seqlen_q_tile P]
   - tensor_scalar_multiply(exp_res, inverse_sum_res) = softmax_res[seqlen_k F, seqlen_q_tile P]
  The tensor_reduce_add and reciprocal arrays are stored for reuse in the backward pass

  5. transpose(softmax_res) = trans_softmax_res[seqlen_q_tile F, seqlen_k P]
  6. matmul_1(lhs=trans_softmax_res, rhs=trans_v, contract=seqlen_v=seqlen_k)
     = attn_result[seqlen_q_tile F, d_head P]


  Some naming convention used:
   - ip_[name] and if_[name] are tile indices returned from nl.arange(partition dim size) and nl.arange(free dim size).
     ip_[name] and if_[name] represent the partition and free indices respectively. Such tile indices are typically used
     to slice a tensor for a Neuron instruction as input/output. We also use these tile indices to build affine expressions
     for IO tensor accesses and predicates/masks.
  """

  # Use q_ref dtype as the intermediate tensor dtype
  # Assume all IO tensors have the same dtype
  kernel_dtype = q_ref.dtype
  mixed_dtype = np.dtype(np.float32) if mixed_precision else kernel_dtype
  assert q_ref.dtype == k_ref.dtype == v_ref.dtype == out_ref.dtype
  assert out_cached_negative_max_ref.dtype == out_cached_sum_reciprocal_ref.dtype == mixed_dtype

  # Shape checking
  bs, nheads, d_head, seqlen = q_ref.shape
  assert tuple(k_ref.shape) == (bs, nheads, d_head, seqlen), 'Input shape mismatch!'
  assert tuple(v_ref.shape) == (bs, nheads, d_head, seqlen), 'Input shape mismatch!'
  assert tuple(q_ref.shape) == (bs, nheads, d_head, seqlen), 'Input shape mismatch!'
  assert tuple(out_ref.shape) == (bs, nheads, d_head, seqlen), 'Output shape mismatch!'
  assert tuple(out_cached_negative_max_ref.shape) == (bs, nheads, nl.tile_size.pmax, seqlen // nl.tile_size.pmax), \
    'out_cached_negative_max_ref mismatch!'
  assert tuple(out_cached_sum_reciprocal_ref.shape) == (bs, nheads, nl.tile_size.pmax, seqlen // nl.tile_size.pmax), \
    'out_cached_sum_reciprocal_ref shape mismatch!'
  if seed_ref is not None:
    assert tuple(seed_ref.shape) == (1,), \
      f"Input seed shape mismatch, got {seed_ref.shape}"
  assert d_head <= 128 or (d_head % 128 == 0), 'd_head must be <= 128 or divisible by 128!'

  # Magic number
  BIG_NUMBER = 9984.0

  # Softmax scaling factor, multiplied onto Q
  softmax_scale = 1.0 / float(d_head ** 0.5)

  # Different batch samples/attention heads have independent attention
  batch_id = nl.program_id(axis=0)
  head_id = nl.program_id(axis=1)

  q_seq_n_tiles, q_seq_tile_size = div_ceil(seqlen, 128), 128
  d_head_n_tiles, d_head_tile_size = div_ceil(d_head, 128), min(d_head, 128)
  v_seq_n_tiles, v_seq_tile_size = div_ceil(seqlen, 128), 128
  if seqlen >= 512:
    k_seq_n_tiles, k_seq_tile_size = seqlen // 512, 512
  else:
    k_seq_n_tiles, k_seq_tile_size = seqlen // 128, 128

  k_seq_v_seq_multipler = k_seq_tile_size // v_seq_tile_size

  ip_qk = nl.arange(d_head_tile_size)[:, None]
  if_q = nl.arange(q_seq_tile_size)[None, :]
  if_k = nl.arange(k_seq_tile_size)[None, :]
  ###################################
  # Step 1. transpose(tensor_v)
  ###################################
  # Buffer for v matrix transposed
  # Pre-fetch and keep it in SBUF throughout different softmax tiles
  trans_v = nl.ndarray((par_dim(v_seq_tile_size), v_seq_n_tiles, d_head), dtype=kernel_dtype)
  for i_d_head_tile in nl.affine_range(d_head_n_tiles):
    for i_v_seq_tile in nl.affine_range(v_seq_n_tiles):
      ip_v = nl.arange(d_head_tile_size)[:, None]
      if_v = nl.arange(v_seq_tile_size)[None, :]
      ip_v_t = nl.arange(v_seq_tile_size)[:, None]
      if_v_t = nl.arange(d_head_tile_size)[None, :]
      v_local = nl.load(
        v_ref[batch_id, head_id, i_d_head_tile * d_head_tile_size + ip_v, i_v_seq_tile * v_seq_tile_size + if_v],
        dtype=kernel_dtype)
      trans_v[ip_v_t, i_v_seq_tile, i_d_head_tile * d_head_tile_size + if_v_t] = nisa.sb_transpose(v_local)

  if dropout_p > 0.0:
    seed_local = nl.ndarray((par_dim(1), 1), buffer=nl.sbuf, dtype=nl.int32)
    seed_local[0, 0] = nl.load(seed_ref[0])
    # TODO: Remove this once the dropout supports scale prob
    dropout_p_local = nl.full((q_seq_tile_size, 1), fill_value=dropout_p, dtype=np.float32)

  # affine_range give the compiler permission to vectorize instructions
  # inside the loop which improves the performance. However, when using the 
  # the dropout we should use sequential_range to avoid setting
  # seed vectorization. TODO: the compiler should avoid vectorizing seed setting
  _range = nl.sequential_range if dropout_p > 0.0 else nl.affine_range
  
  neg_max_res_final = nl.ndarray((par_dim(nl.tile_size.pmax), seqlen // nl.tile_size.pmax), dtype=mixed_dtype)
  sum_divisor = nl.ndarray((par_dim(nl.tile_size.pmax), seqlen // nl.tile_size.pmax), dtype=mixed_dtype)
  for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):
    q_local = nl.ndarray((d_head_n_tiles, par_dim(d_head_tile_size), q_seq_tile_size),
                        dtype=kernel_dtype)
    for i_d_head_tile in nl.affine_range(d_head_n_tiles):
      q_local[i_d_head_tile, ip_qk, if_q] = nl.load(
        q_ref[batch_id, head_id, i_d_head_tile * d_head_tile_size + ip_qk, i_q_seq_tile * q_seq_tile_size + if_q],
        dtype=kernel_dtype) * softmax_scale

    # A SBUF buffer for an independent softmax tile
    qk_res_buf = nl.ndarray((par_dim(q_seq_tile_size), seqlen), buffer=nl.sbuf, dtype=kernel_dtype)
    neg_max_res = nl.full((par_dim(q_seq_tile_size), k_seq_n_tiles), fill_value=3.3895314e+38, buffer=nl.sbuf, dtype=kernel_dtype)
    ip_max = nl.arange(q_seq_tile_size)[:, None]
    if_max = nl.arange(k_seq_n_tiles)[None, :]
    # Loop over LHS free of matmul(lhs=tensor_k, rhs=tensor_q, contract=d_head)
    for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
      k_local = nl.ndarray((d_head_n_tiles, par_dim(d_head_tile_size), k_seq_tile_size), dtype=kernel_dtype)
      for i_d_head_tile in nl.affine_range(d_head_n_tiles):
        k_local[i_d_head_tile, ip_qk, if_k] = nl.load(
          k_ref[batch_id, head_id, i_d_head_tile * d_head_tile_size + ip_qk, i_k_seq_tile * k_seq_tile_size + if_k],
          dtype=kernel_dtype)

      forward_mask = i_q_seq_tile * q_seq_tile_size >= i_k_seq_tile * k_seq_tile_size if use_causal_mask else None
      # PSUM buffer shape: [q_seq_tile_size P, k_seq_tile_size F]
      qk_psum = nl.zeros((par_dim(q_seq_tile_size), k_seq_tile_size),
                         dtype=np.float32, buffer=nl.psum)

      # Tensor indices for accessing qk result in k_seq_tile_size
      ip_head = nl.arange(d_head_tile_size)[:, None]
      ip_qk = nl.arange(q_seq_tile_size)[:, None]
      if_qk = nl.arange(k_seq_tile_size)[None, :]

      # Loop over contraction dim of QK matmul
      for i_d_head_tile in nl.affine_range(d_head_n_tiles):  # indent = 6
        ##############################################################
        # Step 2. matmul(lhs=tensor_k, rhs=tensor_q, contract=d_head)
        ##############################################################
        qk_psum[ip_qk, if_qk] += nisa.nc_matmul(q_local[i_d_head_tile, ip_head, if_q],
                                                k_local[i_d_head_tile, ip_head, if_k],
                                                mask=forward_mask)

      ###################################
      # Step 3. Apply optional causal mask
      ###################################
      # causes NAN
      if use_causal_mask:
        # Magic number -9984.0 to replace -inf similar to what Tensorizer uses
        qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk] = nisa.affine_select(
          pred=(i_q_seq_tile * q_seq_tile_size + ip_qk >= i_k_seq_tile * k_seq_tile_size + if_qk),
          on_true_tile=qk_psum[ip_qk, if_qk], on_false_value=-BIG_NUMBER, dtype=kernel_dtype,
          # mask=forward_mask # causes NAN issue
         )
      else:
        # Simply send psum result back to sbuf
        qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk] = \
          nl.copy(qk_psum[ip_qk, if_qk], dtype=kernel_dtype)

      ###################################
      # Step 4. Softmax
      ###################################
      neg_max_res[ip_qk, i_k_seq_tile] = nisa.reduce(
        np.max, data=qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk],
        mask=forward_mask,
        axis=(1,), dtype=kernel_dtype, negate=True)

    neg_max_res_final[ip_qk, i_q_seq_tile] = nisa.reduce(
      np.min, data=neg_max_res[ip_max, if_max],
      axis=(1,), dtype=kernel_dtype, negate=False)

    ip_softmax = nl.arange(q_seq_tile_size)[:, None]
    if_softmax = nl.arange(k_seq_tile_size)[None, :]
    softmax_numerator = nl.ndarray((par_dim(q_seq_tile_size), seqlen), dtype=mixed_dtype)
    sum_res_partial = nl.zeros((par_dim(q_seq_tile_size), k_seq_n_tiles), buffer=nl.sbuf, dtype=mixed_dtype)
    for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
      forward_mask = i_q_seq_tile * q_seq_tile_size >= i_k_seq_tile * k_seq_tile_size if use_causal_mask else None      
      softmax_numerator[ip_softmax, i_k_seq_tile * k_seq_tile_size + if_softmax] = \
          nisa.activation(np.exp,
                          data=qk_res_buf[ip_softmax, i_k_seq_tile * k_seq_tile_size + if_softmax],
                          bias=neg_max_res_final[ip_softmax, i_q_seq_tile], scale=1.0,
                          mask=forward_mask)

      sum_res_partial[ip_softmax, i_k_seq_tile] = \
          nisa.reduce(np.add, data=softmax_numerator[ip_softmax, i_k_seq_tile * k_seq_tile_size + if_softmax],
                      axis=(1,), dtype=mixed_dtype, mask=forward_mask)

    sum_res = nisa.reduce(np.add, data=sum_res_partial[ip_softmax, if_max], axis=(1,), dtype=mixed_dtype)
    sum_reciprocal = 1.0 / sum_res
    sum_divisor[ip_softmax, i_q_seq_tile] = nl.copy(sum_reciprocal, dtype=mixed_dtype)

    # Loop over matmul_1 RHS free
    for i_d_head_tile in nl.affine_range(d_head_n_tiles):
      # Result psum buffer has the hidden dim as P
      attn_res_psum = nl.zeros((par_dim(d_head_tile_size), q_seq_tile_size),
                               dtype=np.float32, buffer=nl.psum)
      ip_out = nl.arange(d_head_tile_size)[:, None]
      if_out = nl.arange(q_seq_tile_size)[None, :]
      for i_k_seq_tile in _range(k_seq_n_tiles):
        forward_mask = i_q_seq_tile * q_seq_tile_size >= i_k_seq_tile * k_seq_tile_size if use_causal_mask else None
        softmax_y = nl.ndarray((par_dim(q_seq_tile_size), k_seq_tile_size), dtype=kernel_dtype, buffer=nl.sbuf) 
        softmax_y[ip_softmax, if_softmax] = nl.multiply(softmax_numerator[ip_softmax, i_k_seq_tile * k_seq_tile_size + if_softmax],
                                                        sum_divisor[ip_softmax, i_q_seq_tile],
                                                        mask=forward_mask)
        #####################################################################
        # Dropout 
        #####################################################################      
        if dropout_p > 0.0:
          offset = i_k_seq_tile + i_q_seq_tile * k_seq_n_tiles \
                    + head_id * k_seq_n_tiles * q_seq_n_tiles \
                    + batch_id * nl.num_programs(1) * k_seq_n_tiles * q_seq_n_tiles
          offset_seed = nl.add(seed_local[0, 0], offset, mask=forward_mask)
          nisa.random_seed(seed=offset_seed, mask=forward_mask)
          softmax_y[ip_softmax, if_softmax] = nl.dropout(softmax_y[ip_softmax, if_softmax], rate=dropout_p_local[ip_qk, 0], mask=forward_mask)
          softmax_y[ip_softmax, if_softmax] = nl.multiply(softmax_y[ip_softmax, if_softmax], 1 / (1 - dropout_p), mask=forward_mask)

        for i_v_seq_tile in nl.affine_range(k_seq_v_seq_multipler):
          ######################################################################
          # Step 6. matmul_1(lhs=trans_softmax_res, rhs=trans_v, contract=seqlen_v=seqlen_k)
          ######################################################################
          ip_v_t = nl.arange(v_seq_tile_size)[:, None]
          if_v_t = nl.arange(d_head_tile_size)[None, :]
          if_softmax = nl.arange(v_seq_tile_size)[None, :]
          trans_softmax_res = nisa.sb_transpose(
            softmax_y[ip_softmax, i_v_seq_tile * v_seq_tile_size + if_softmax],
            mask=forward_mask)
          attn_res_psum[ip_out, if_out] += \
            nisa.nc_matmul(rhs=trans_softmax_res,
                           lhs=trans_v[ip_v_t, i_k_seq_tile * k_seq_v_seq_multipler + i_v_seq_tile, i_d_head_tile * d_head_tile_size + if_v_t],
                           mask=forward_mask)
      nl.store(
        out_ref[batch_id, head_id, i_d_head_tile * d_head_tile_size + ip_out, i_q_seq_tile * q_seq_tile_size + if_out],
        value=attn_res_psum[ip_out, if_out])

  # Save softmax partials for reuse in the backward pass
  ip_softmax = nl.arange(nl.tile_size.pmax)[:, None]
  if_softmax = nl.arange(seqlen // nl.tile_size.pmax)[None, :]
  nl.store(
    dst=out_cached_negative_max_ref[batch_id, head_id, ip_max, if_softmax],
    value=neg_max_res_final[ip_softmax, if_softmax],
  )
  nl.store(
    dst=out_cached_sum_reciprocal_ref[batch_id, head_id, ip_max, if_softmax],
    value=sum_divisor[ip_softmax, if_softmax],
  )

def fused_self_attn_bwd(
    q_ref, k_ref, v_ref,
    dy_ref,
    out_dq_ref, out_dk_ref, out_dv_ref,
    use_causal_mask=False,
    mixed_precision=False,
):
  """
  Fused self attention backward kernel. Compute the backward gradients.

  IO tensor layouts:
   - q_ref: shape (bs, nheads, head_size, seq)
   - k_ref: shape (bs, nheads, head_size, seq)
   - v_ref: shape (bs, nheads, head_size, seq)
   - dy_ref: shape (bs, nheads, head_size, seq)
   - out_dq_ref: shape (bs, nheads, head_size, seq)
   - out_dk_ref: shape (bs, nheads, head_size, seq)
   - out_dv_ref: shape (bs, nheads, head_size, seq)

  Detailed steps:
    1. Recompute (softmax(Q@K^T))
      1.1 Q@K^T
      1.2 Scale the QK score
      1.3 Apply causal mask
      1.4 softmax
    2. Compute the gradients of y = score @ V with respect to the loss

    3. Compute the gradients of y = softmax(x)

    4. Compute the gradients of Q@K^T
      4.1 Compute dQ
      4.2 Compute dK
  """

  # Use q_ref dtype as the intermediate tensor dtype
  # Assume all IO tensors have the same dtype
  kernel_dtype = q_ref.dtype
  tensor_acc_dtype = np.dtype(np.float32) if mixed_precision else kernel_dtype

  assert q_ref.dtype == k_ref.dtype == v_ref.dtype == dy_ref.dtype \
         == out_dq_ref.dtype == out_dk_ref.dtype == out_dv_ref.dtype

  # Shape checking
  bs, nheads, d_head, seqlen = q_ref.shape
  assert tuple(k_ref.shape) == (bs, nheads, d_head, seqlen), \
    f"Input K shape mismatch, got {k_ref.shape}"
  assert tuple(v_ref.shape) == (bs, nheads, d_head, seqlen), \
    f"Input V shape mismatch, got {v_ref.shape}"
  assert tuple(dy_ref.shape) == (bs, nheads, d_head, seqlen), \
    f"Input dy shape mismatch, got {dy_ref.shape}"

  assert tuple(out_dq_ref.shape) == (bs, nheads, d_head, seqlen), \
    f"Output dQ shape mismatch, got {out_dq_ref.shape}"
  assert tuple(out_dk_ref.shape) == (bs, nheads, d_head, seqlen), \
    f"Output dK shape mismatch, got {out_dk_ref.shape}"
  assert tuple(out_dv_ref.shape) == (bs, nheads, d_head, seqlen), \
    f"Output dV shape mismatch, got {out_dv_ref.shape}"

  # FIXME: Add masking for different seqlen values.
  assert seqlen % 128 == 0, \
    f"Input sequence length must be divisible by 128, got {seqlen}"

  # Softmax scaling factor, multiplied onto Q
  softmax_scale = 1.0 / float(d_head ** 0.5)

  # Different batch samples/attention heads have independent attention
  batch_id = nl.program_id(axis=0)
  head_id = nl.program_id(axis=1)

  q_seq_n_tiles, q_seq_tile_size = div_ceil(seqlen, 128), 128
  d_head_n_tiles, d_head_tile_size = div_ceil(d_head, 128), 128

  if seqlen >= 512:
    k_seq_n_tiles, k_seq_tile_size = seqlen // 512, 512
    v_seq_n_tiles, v_seq_tile_size = seqlen // 512, 512
  else:
    k_seq_n_tiles, k_seq_tile_size = seqlen // 128, 128
    v_seq_n_tiles, v_seq_tile_size = seqlen // 128, 128

  k_seq_n_tiles_backward, k_seq_tile_size_backward = seqlen // 128, 128
  k_seq_fwd_bwd_tile_multipler = k_seq_tile_size // k_seq_tile_size_backward

  ip_qk = nl.arange(d_head_tile_size)[:, None]
  if_q = nl.arange(q_seq_tile_size)[None, :]
  if_k = nl.arange(k_seq_tile_size)[None, :]

  # Prefetch dy

  # If head size is not a multiple of 128, we use 128 for tile size but mask out
  # computation as needed for numerical correctness. Note not all computation
  # is masked out, so we initialize relevant tensors to 0 to maintain numerical
  # correctness when head size is not a multiple of 128.
  dy_local = nl.zeros((q_seq_n_tiles, d_head_n_tiles, par_dim(d_head_tile_size), q_seq_tile_size), dtype=kernel_dtype)
  for i_d_head_tile in nl.affine_range(d_head_n_tiles):
    for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):
      ip_qk_mask = i_d_head_tile * d_head_tile_size + ip_qk < d_head

      dy_local[i_q_seq_tile, i_d_head_tile, ip_qk, if_q] = nl.load(
        dy_ref[batch_id, head_id, i_d_head_tile * d_head_tile_size + ip_qk, i_q_seq_tile * q_seq_tile_size + if_q],
        dtype=kernel_dtype,
        mask=ip_qk_mask)

  # Prefetch V
  v_local = nl.zeros((v_seq_n_tiles, d_head_n_tiles, par_dim(d_head_tile_size), v_seq_tile_size), dtype=kernel_dtype)
  for i_d_head_tile in nl.affine_range(d_head_n_tiles):
    for i_v_seq_tile in nl.affine_range(v_seq_n_tiles):
      ip_v = nl.arange(d_head_tile_size)[:, None]
      if_v = nl.arange(v_seq_tile_size)[None, :]

      ip_v_mask = i_d_head_tile * d_head_tile_size + ip_v < d_head

      v_local[i_v_seq_tile, i_d_head_tile, ip_v, if_v] = nl.load(
        v_ref[batch_id, head_id, i_d_head_tile * d_head_tile_size + ip_v, i_v_seq_tile * v_seq_tile_size + if_v],
        dtype=kernel_dtype,
        mask=ip_v_mask)

  # Prefetch Q
  q_local = nl.zeros((q_seq_n_tiles, d_head_n_tiles, par_dim(d_head_tile_size), q_seq_tile_size), dtype=kernel_dtype)
  for i_d_head_tile in nl.affine_range(d_head_n_tiles):
    for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):
      ##############################################################
      # Step 1.3 Scale the score. Here we multiply into q matrix directly,
      # which is mathematically equivalent
      ##############################################################
      ip_qk_mask = i_d_head_tile * d_head_tile_size + ip_qk < d_head

      q_local[i_q_seq_tile, i_d_head_tile, ip_qk, if_q] = nl.load(
        q_ref[batch_id, head_id, i_d_head_tile * d_head_tile_size + ip_qk, i_q_seq_tile * q_seq_tile_size + if_q],
        dtype=kernel_dtype,
        mask=ip_qk_mask) * softmax_scale

  # Prefetch K
  k_local = nl.zeros((k_seq_n_tiles, d_head_n_tiles, par_dim(d_head_tile_size), k_seq_tile_size), dtype=kernel_dtype)
  transposed_k_local = nl.zeros((k_seq_n_tiles_backward, d_head_n_tiles, par_dim(k_seq_tile_size_backward), d_head_tile_size), dtype=kernel_dtype)
  for i_d_head_tile in nl.affine_range(d_head_n_tiles):
    for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
      ip_qk_mask = i_d_head_tile * d_head_tile_size + ip_qk < d_head

      k_local[i_k_seq_tile, i_d_head_tile, ip_qk, if_k] = nl.load(
        k_ref[batch_id, head_id, i_d_head_tile * d_head_tile_size + ip_qk, i_k_seq_tile * k_seq_tile_size + if_k],
        dtype=kernel_dtype,
        mask=ip_qk_mask)

      ##############################################################
      # Prefetch k transpose for the backward too
      ##############################################################
      if_k_backward = nl.arange(k_seq_tile_size_backward)[None, :]
      ip_k_backward = nl.arange(k_seq_tile_size_backward)[:, None]
      if_d_head = nl.arange(d_head_tile_size)[None, :]
      for i_k_seq_tile_backward in nl.affine_range(k_seq_fwd_bwd_tile_multipler):
        transposed_k_local[i_k_seq_tile * k_seq_fwd_bwd_tile_multipler + i_k_seq_tile_backward, i_d_head_tile, ip_k_backward, if_d_head] = \
          nisa.sb_transpose(k_local[i_k_seq_tile, i_d_head_tile, ip_qk,
                                    i_k_seq_tile_backward * k_seq_tile_size_backward + if_k_backward])


  dv_local_reduced = nl.zeros((k_seq_n_tiles, d_head_n_tiles, par_dim(d_head_tile_size), k_seq_tile_size),
                                dtype=tensor_acc_dtype)
  dk_local_reduced = nl.zeros((k_seq_n_tiles, d_head_n_tiles, par_dim(d_head_tile_size), k_seq_tile_size),
                                dtype=tensor_acc_dtype)
  for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):
    # A SBUF buffer for an independent softmax tile
    qk_res_buf = nl.ndarray((par_dim(q_seq_tile_size), seqlen), buffer=nl.sbuf, dtype=kernel_dtype)
    neg_max_res = nl.full((par_dim(q_seq_tile_size), k_seq_n_tiles), fill_value=np.inf, buffer=nl.sbuf, dtype=kernel_dtype)
    # Loop over LHS free of matmul(lhs=tensor_k, rhs=tensor_q, contract=d_head)
    for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
      forward_mask = i_q_seq_tile * q_seq_tile_size >= i_k_seq_tile * k_seq_tile_size if use_causal_mask else None
      # PSUM buffer shape: [q_seq_tile_size P, k_seq_tile_size F]
      qk_psum = nl.zeros((par_dim(q_seq_tile_size), k_seq_tile_size),
                         dtype=np.float32, buffer=nl.psum)

      # Tensor indices for accessing qk result in k_seq_tile_size
      ip_qk = nl.arange(q_seq_tile_size)[:, None]
      if_qk = nl.arange(k_seq_tile_size)[None, :]

      # Loop over contraction dim of QK matmul
      for i_d_head_tile in nl.affine_range(d_head_n_tiles):
        ##############################################################
        # Step 1.1 Compute Q@K^T, with matmul(lhs=tensor_k, rhs=tensor_q, contract=d_head)
        ##############################################################
        qk_psum[ip_qk, if_qk] += nisa.nc_matmul(q_local[i_q_seq_tile, i_d_head_tile, ip_qk, if_q],
                                                k_local[i_k_seq_tile, i_d_head_tile, ip_qk, if_k],
                                                mask=forward_mask)

      ###################################
      # Step 1.2. Apply optional causal mask
      ###################################
      if use_causal_mask:
        # Magic number -9984.0 to replace -inf similar to what Tensorizer uses
        qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk] = nisa.affine_select(
          pred=(i_q_seq_tile * q_seq_tile_size + ip_qk >= i_k_seq_tile * k_seq_tile_size + if_qk),
          on_true_tile=qk_psum[ip_qk, if_qk], on_false_value=-9984.0, dtype=kernel_dtype,
          mask=forward_mask)
      else:
        # Simply send psum result back to sbuf
        qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk] = \
          nl.copy(qk_psum[ip_qk, if_qk], dtype=kernel_dtype)

      #######################################################
      # Step 1.4 Recompute the softmax in the forward
      #######################################################
      neg_max_res[ip_qk, i_k_seq_tile] = nisa.reduce(
        np.max, data=qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk],
        mask=forward_mask,
        axis=(1,), dtype=kernel_dtype, negate=True)

    if_max = nl.arange(k_seq_n_tiles)[None, :]
    neg_max_res_final = nl.ndarray((par_dim(q_seq_tile_size), 1), dtype=kernel_dtype)
    neg_max_res_final[ip_qk, 0] = nisa.reduce(
      np.min, data=neg_max_res[ip_qk, if_max],
      axis=(1,), dtype=kernel_dtype, negate=False)

    ip_softmax = nl.arange(q_seq_tile_size)[:, None]
    ip_sum_res = nl.arange(q_seq_tile_size)[:, None]

    softmax_numerator = nl.ndarray((par_dim(q_seq_tile_size), seqlen), dtype=kernel_dtype)
    sum_divisor = nl.ndarray((par_dim(q_seq_tile_size), 1), dtype=kernel_dtype)

    if_softmax = nl.arange(k_seq_tile_size)[None, :]
    sum_res_partial = nl.zeros((par_dim(q_seq_tile_size), k_seq_n_tiles), buffer=nl.sbuf, dtype=tensor_acc_dtype)
    for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
      forward_mask = i_q_seq_tile * q_seq_tile_size >= i_k_seq_tile * k_seq_tile_size if use_causal_mask else None
      softmax_numerator[ip_softmax, i_k_seq_tile * k_seq_tile_size + if_softmax] = \
          nisa.activation(np.exp,
                          data=qk_res_buf[ip_softmax, i_k_seq_tile * k_seq_tile_size + if_softmax],
                          bias=neg_max_res_final[ip_softmax, 0], scale=1.0,
                          mask=forward_mask)

      sum_res_partial[ip_softmax, i_k_seq_tile] = \
          nisa.reduce(np.add, data=softmax_numerator[ip_softmax, i_k_seq_tile * k_seq_tile_size + if_softmax],
                      axis=(1,), dtype=tensor_acc_dtype, mask=forward_mask)

    sum_res = nisa.reduce(np.add, data=sum_res_partial[ip_softmax, if_max], axis=(1,), dtype=tensor_acc_dtype)
    sum_reciprocal = 1.0 / sum_res
    sum_divisor[ip_sum_res, 0] = nl.copy(sum_reciprocal, dtype=kernel_dtype)

    softmax_y_times_dy_sum_partial = nl.zeros((par_dim(q_seq_tile_size), k_seq_n_tiles),
                                              dtype=tensor_acc_dtype, buffer=nl.sbuf)
    softmax_dy = nl.ndarray((par_dim(q_seq_tile_size), seqlen), dtype=kernel_dtype, buffer=nl.sbuf)
    softmax_y = nl.ndarray((par_dim(q_seq_tile_size), seqlen), dtype=kernel_dtype, buffer=nl.sbuf)
    for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
      ip_v = nl.arange(d_head_tile_size)[:, None]
      if_v = nl.arange(k_seq_tile_size)[None, :]
      backward_mask = i_q_seq_tile * q_seq_tile_size >= i_k_seq_tile * k_seq_tile_size if use_causal_mask else None
      softmax_y[ip_softmax, i_k_seq_tile * k_seq_tile_size + if_v] = \
          nl.multiply(softmax_numerator[ip_softmax, i_k_seq_tile * k_seq_tile_size + if_v],
                        sum_divisor[ip_softmax, 0],
                        mask=backward_mask)

      #####################################################################
      # Step 2.1 Calculate the backward gradients dL/dV, where y=softmax@V
      # in value projection with matmul(LHS=softmax, RHS=dy)
      #####################################################################
      for i_d_head_tile in nl.affine_range(d_head_n_tiles):
        ip_dv = nl.arange(d_head_tile_size)[:, None]
        if_dv = nl.arange(k_seq_tile_size)[None, :]
        trans_dy = nisa.sb_transpose(dy_local[i_q_seq_tile, i_d_head_tile, ip_v, if_q],
                                     mask=backward_mask)
        dv_psum = nisa.nc_matmul(trans_dy,
                                 softmax_y[ip_softmax, i_k_seq_tile * k_seq_tile_size + if_dv],
                                 mask=backward_mask)

        ip_dv_mask = i_d_head_tile * d_head_tile_size + ip_dv < d_head

        dv_local_reduced[i_k_seq_tile, i_d_head_tile, ip_dv, if_dv] = nl.loop_reduce(
                      dv_psum, op=np.add, loop_indices=(i_q_seq_tile,),
                      dtype=tensor_acc_dtype,
                      mask=(backward_mask & ip_dv_mask if backward_mask else ip_dv_mask))


      #####################################################################
      # Step 2.2 Calculate the backward gradients dL/dsoftmax, where y=softmax@V
      # in value projection with matmul(LHS=v, RHS=dy)
      #####################################################################
      softmax_dy_psum = nl.zeros((par_dim(q_seq_tile_size), k_seq_tile_size),
                                 dtype=np.float32, buffer=nl.psum)
      for i_d_head_tile in nl.affine_range(d_head_n_tiles):
        softmax_dy_psum[ip_softmax, if_v] += \
          nisa.nc_matmul(dy_local[i_q_seq_tile, i_d_head_tile, ip_v, if_q],
                         v_local[i_k_seq_tile, i_d_head_tile, ip_v, if_v],
                         mask=backward_mask)

      softmax_dy[ip_softmax, i_k_seq_tile * k_seq_tile_size + if_v] = \
        nl.copy(softmax_dy_psum[ip_softmax, if_v], dtype=kernel_dtype,
                  mask=backward_mask)

      #####################################################################
      # Step 3 Calculate the softmax backward gradients dL/dx, where y=softmax(x)
      # dL/dx = y * (dL/dy - sum(dL/dy * y)), where y = softmax(x)
      #####################################################################
      softmax_y_times_dy = nl.multiply(softmax_dy[ip_softmax, i_k_seq_tile * k_seq_tile_size + if_v],
                                       softmax_y[ip_softmax, i_k_seq_tile * k_seq_tile_size + if_v],
                                       dtype=kernel_dtype,
                                       mask=backward_mask)
      softmax_y_times_dy_sum_partial[ip_softmax, i_k_seq_tile] = \
        nisa.reduce(np.add, data=softmax_y_times_dy, axis=(1,), dtype=tensor_acc_dtype,
                    mask=backward_mask)

    softmax_y_times_dy_sum = nl.ndarray((par_dim(q_seq_tile_size), 1), dtype=tensor_acc_dtype)
    softmax_y_times_dy_sum[ip_softmax, 0] =  \
      nisa.reduce(np.add,
                  data=softmax_y_times_dy_sum_partial[ip_softmax, nl.arange(k_seq_n_tiles)[None, :]],
                  axis=(1, ), dtype=tensor_acc_dtype)

    if_k = nl.arange(k_seq_tile_size)[None, :]
    softmax_dx_local = nl.ndarray((k_seq_n_tiles, par_dim(q_seq_tile_size), k_seq_tile_size),
                                  dtype=kernel_dtype, buffer=nl.sbuf)
    transposed_softmax_dx_local = nl.ndarray((k_seq_n_tiles_backward, par_dim(k_seq_tile_size_backward), q_seq_tile_size),
                                             dtype=kernel_dtype, buffer=nl.sbuf)
    for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
      backward_mask = i_q_seq_tile * q_seq_tile_size >= i_k_seq_tile * k_seq_tile_size if use_causal_mask else None
      # y * (dL/dy - sum(dL/dy * y))
      softmax_dx_local[i_k_seq_tile, ip_softmax, if_k] = \
        nisa.tensor_scalar(data=softmax_dy[ip_softmax, i_k_seq_tile * k_seq_tile_size + if_k],
                           op0=np.subtract,
                           operand0=softmax_y_times_dy_sum[ip_softmax, 0],
                           op1=np.multiply,
                           operand1=softmax_y[ip_softmax, i_k_seq_tile * k_seq_tile_size + if_k],
                           mask=backward_mask)

    #####################################################################
    # Step 4.2 Calculate dK, with matmul(LHS=softmax_dx, RHS=Q)
    #####################################################################
    ip_trans_q = nl.arange(d_head_tile_size)[:, None]
    if_trans_q = nl.arange(q_seq_tile_size)[None, :]
    if_softmax_dx = nl.arange(k_seq_tile_size)[None, :]
    ip_dk = nl.arange(d_head_tile_size)[:, None]
    for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
        backward_mask = i_q_seq_tile * q_seq_tile_size >= i_k_seq_tile * k_seq_tile_size if use_causal_mask else None
        for i_d_head_tile in nl.affine_range(d_head_n_tiles):
          trans_q_local = nisa.sb_transpose(q_local[i_q_seq_tile, i_d_head_tile, ip_trans_q, if_trans_q],
                                            mask=backward_mask)
          dk_psum = nisa.nc_matmul(
                      trans_q_local,
                      softmax_dx_local[i_k_seq_tile, ip_softmax, if_softmax_dx],
                      mask=backward_mask)

          ip_dk_mask = i_d_head_tile * d_head_tile_size + ip_dk < d_head

          dk_local_reduced[i_k_seq_tile, i_d_head_tile, ip_dk, if_softmax_dx] = nl.loop_reduce(
            dk_psum, op=np.add, loop_indices=(i_q_seq_tile,),
            dtype=tensor_acc_dtype,
            mask=(backward_mask & ip_dk_mask if backward_mask else ip_dk_mask))

        # Transpose softmax_dx early to avoid the tranpose under contract dimension of dQ
        ip_k = nl.arange(k_seq_tile_size_backward)[:, None]
        if_k = nl.arange(k_seq_tile_size_backward)[None, :]
        for i_k_seq_tile_backward in nl.affine_range(k_seq_fwd_bwd_tile_multipler):
          transposed_softmax_dx_local[i_k_seq_tile * k_seq_fwd_bwd_tile_multipler + i_k_seq_tile_backward, ip_k, if_trans_q] = \
            nisa.sb_transpose(softmax_dx_local[i_k_seq_tile, ip_softmax,
                                               i_k_seq_tile_backward * k_seq_tile_size_backward + if_k],
                              mask=backward_mask)

    #####################################################################
    # Step 4.1 Calculate dQ
    #####################################################################
    ip_k = nl.arange(d_head_tile_size)[:, None]
    if_k = nl.arange(k_seq_tile_size_backward)[None, :]
    ip_dq = nl.arange(d_head_tile_size)[:, None]
    if_dq = nl.arange(q_seq_tile_size)[None, :]
    ip_transposed_k = nl.arange(k_seq_tile_size_backward)[:, None]
    for i_d_head_tile in nl.affine_range(d_head_n_tiles):
      dq_psum = nl.zeros((par_dim(d_head_tile_size), q_seq_tile_size),
                         dtype=np.float32, buffer=nl.psum)
      for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
        backward_mask = i_q_seq_tile * q_seq_tile_size >= i_k_seq_tile * k_seq_tile_size if use_causal_mask else None
        for i_k_seq_tile_backward in nl.affine_range(k_seq_fwd_bwd_tile_multipler):
          dq_psum[ip_dq, if_dq] += nisa.nc_matmul(transposed_k_local[i_k_seq_tile * k_seq_fwd_bwd_tile_multipler + i_k_seq_tile_backward,
                                                                     i_d_head_tile, ip_transposed_k, if_dq],
                                                  transposed_softmax_dx_local[i_k_seq_tile * k_seq_fwd_bwd_tile_multipler + i_k_seq_tile_backward,
                                                                              ip_transposed_k, if_dq],
                                                  mask=backward_mask)

      dq_local = nl.multiply(dq_psum[ip_dq, if_dq], softmax_scale, dtype=kernel_dtype)

      ip_dq_mask = i_d_head_tile * d_head_tile_size + ip_dq < d_head

      nl.store(
        out_dq_ref[batch_id, head_id, i_d_head_tile * d_head_tile_size + ip_dq, i_q_seq_tile * q_seq_tile_size + if_dq],
        value=dq_local,
        mask=ip_dq_mask
      )

  #####################################################################
  # Store dK, dV (at end to maintain loop fusion)
  #####################################################################
  for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
    for i_d_head_tile in nl.affine_range(d_head_n_tiles):
      ip_dkv = nl.arange(d_head_tile_size)[:, None]
      if_dkv = nl.arange(k_seq_tile_size)[None, :]


      ip_dkv_mask = i_d_head_tile * d_head_tile_size + ip_dkv < d_head

      nl.store(
        out_dv_ref[batch_id, head_id,
                    i_d_head_tile * d_head_tile_size + ip_dv,
                    i_k_seq_tile * k_seq_tile_size + if_dv],
        value=dv_local_reduced[i_k_seq_tile, i_d_head_tile, ip_dkv, if_dkv],
        mask=ip_dkv_mask
      )

      nl.store(
        out_dk_ref[batch_id, head_id,
                    i_d_head_tile * d_head_tile_size + ip_dk,
                    i_k_seq_tile * k_seq_tile_size + if_softmax_dx],
        value=dk_local_reduced[i_k_seq_tile, i_d_head_tile, ip_dkv, if_dkv],
        mask=ip_dkv_mask
      )


@trace
def _flash_attn_bwd_core(
  q_local, k_local, transposed_k_local, v_local, dy_local,
  dk_psum, dv_psum, dq_local_reduced,
  softmax_exp_bias, dy_o_sum,
  local_i_q_seq_tile, local_i_k_seq_tile,
  seqlen, d_head, 
  use_causal_mask,
  kernel_dtype, mixed_dtype,
  softmax_scale,
  seed_local, dropout_p, dropout_p_local,
  global_i_q_seq_tile = None,
  global_i_k_seq_tile = None,
):
  """
  The flash backward core funciton to calculate the gradients of Q, K and V
  of the given tiles. The result will be accumulated into the dk, dv, dq psum
  """
  q_seq_n_tiles, q_seq_tile_size = div_ceil(seqlen, 128), 128
  d_head_n_tiles, d_head_tile_size = div_ceil(d_head, 128), min(d_head, 128)
  if seqlen >= 512:
    k_seq_n_tiles, k_seq_tile_size = seqlen // 512, 512
  else:
    k_seq_n_tiles, k_seq_tile_size = seqlen // 128, 128
  k_seq_n_tiles_backward, k_seq_tile_size_backward = seqlen // 128, 128
  k_seq_fwd_bwd_tile_multipler = k_seq_tile_size // k_seq_tile_size_backward

  if global_i_q_seq_tile is None:
    global_i_q_seq_tile = local_i_q_seq_tile
    global_i_k_seq_tile = local_i_k_seq_tile

  mask = global_i_q_seq_tile * q_seq_tile_size >= global_i_k_seq_tile * k_seq_tile_size if use_causal_mask else None
  # PSUM buffer shape: [q_seq_tile_size P, k_seq_tile_size F]
  qk_psum = nl.zeros((par_dim(q_seq_tile_size), k_seq_tile_size),
                      dtype=np.float32, buffer=nl.psum)
  qk_res_buf = nl.ndarray((par_dim(q_seq_tile_size), k_seq_tile_size), buffer=nl.sbuf, dtype=kernel_dtype)
  
  batch_id = nl.program_id(axis=0)
  head_id = nl.program_id(axis=1)
  # Tensor indices for accessing qk result in k_seq_tile_size
  if_q = nl.arange(q_seq_tile_size)[None, :]
  ip_qk = nl.arange(q_seq_tile_size)[:, None]
  if_qk = nl.arange(k_seq_tile_size)[None, :]
  if_k = nl.arange(k_seq_tile_size)[None, :]

  # Loop over contraction dim of QK matmul
  for i_d_head_tile in nl.affine_range(d_head_n_tiles):
    ##############################################################
    # Step 2.1 Compute Q@K^T, with matmul(lhs=tensor_k, rhs=tensor_q, contract=d_head)
    ##############################################################
    qk_psum[ip_qk, if_qk] += nisa.nc_matmul(q_local[i_d_head_tile, ip_qk, if_q],
                                            k_local[i_d_head_tile, ip_qk, if_k],
                                            mask=mask)

  ######################################
  # Step 2.2. Apply optional causal mask
  ######################################
  if use_causal_mask:
    # Magic number -9984.0 to replace -inf similar to what Tensorizer uses
    qk_res_buf[ip_qk, if_qk] = nisa.affine_select(
      pred=(global_i_q_seq_tile * q_seq_tile_size + ip_qk >= global_i_k_seq_tile * k_seq_tile_size + if_qk),
      on_true_tile=qk_psum[ip_qk, if_qk], on_false_value=-9984.0, dtype=mixed_dtype,
      mask=mask)
  else:
    # Simply send psum result back to sbuf
    qk_res_buf[ip_qk, if_qk] = \
      nl.copy(qk_psum[ip_qk, if_qk], dtype=mixed_dtype)

  softmax_y = nl.ndarray((par_dim(q_seq_tile_size), k_seq_tile_size), dtype=kernel_dtype, buffer=nl.sbuf)
  softmax_y[ip_qk, if_qk] = nisa.activation(np.exp,
                                            data=qk_res_buf[ip_qk, if_qk],
                                            bias=softmax_exp_bias[local_i_q_seq_tile, ip_qk, 0],
                                            scale=1.0,
                                            mask=mask)
  #####################################################################
  # Dropout 
  #####################################################################      
  if dropout_p > 0.0:
    offset = global_i_k_seq_tile + global_i_q_seq_tile * k_seq_n_tiles \
              + head_id * k_seq_n_tiles * q_seq_n_tiles \
              + batch_id * nl.num_programs(1) * k_seq_n_tiles * q_seq_n_tiles
    offset_seed = nl.add(seed_local[0, 0], offset, mask=mask)
    nisa.random_seed(seed=offset_seed, mask=mask)
    softmax_y[ip_qk, if_qk] = nl.dropout(softmax_y[ip_qk, if_qk], rate=dropout_p_local[ip_qk, 0], mask=mask)
    softmax_y[ip_qk, if_qk] = nl.multiply(softmax_y[ip_qk, if_qk], 1 / (1 - dropout_p), mask=mask)

  #####################################################################
  # Step 3.1 Calculate the backward gradients dL/dV, where y=softmax@V
  # in value projection with matmul(LHS=softmax, RHS=dy)
  #####################################################################
  for i_d_head_tile in nl.affine_range(d_head_n_tiles):
    ip_dv = nl.arange(d_head_tile_size)[:, None]
    if_dv = nl.arange(k_seq_tile_size)[None, :]
    if_trans_dy = nl.arange(q_seq_tile_size)[None, :]
    trans_dy = nisa.sb_transpose(dy_local[i_d_head_tile, ip_dv, if_trans_dy],
                                  mask=mask)
    dv_psum[i_d_head_tile, ip_dv, if_dv] += \
      nisa.nc_matmul(trans_dy, softmax_y[ip_qk, if_qk], mask=mask)

  #####################################################################
  # Step 3.2 Calculate the backward gradients dL/dsoftmax, where y=softmax@V
  # in value projection with matmul(LHS=v, RHS=dy)
  #####################################################################
  softmax_dy_psum = nl.zeros((par_dim(q_seq_tile_size), k_seq_tile_size),
                              dtype=np.float32, buffer=nl.psum)
  for i_d_head_tile in nl.affine_range(d_head_n_tiles):
    ip_softmax_dy = nl.arange(d_head_tile_size)[:, None]
    if_dy = nl.arange(q_seq_tile_size)[None, :]
    softmax_dy_psum[ip_qk, if_qk] += \
      nisa.nc_matmul(dy_local[i_d_head_tile, ip_softmax_dy, if_dy],
                      v_local[i_d_head_tile, ip_softmax_dy, if_qk],
                      mask=mask)

  softmax_dy = nl.ndarray((par_dim(q_seq_tile_size), k_seq_tile_size), dtype=kernel_dtype, buffer=nl.sbuf)
  softmax_dy[ip_qk, if_qk] = nl.copy(softmax_dy_psum[ip_qk, if_qk], dtype=kernel_dtype,
                                      mask=mask)

  #####################################################################
  # Step 4 Calculate the softmax backward gradients dL/dx, where y=softmax(x)
  # dL/dx = y * (dL/dy - rowsum(dO_O)), where y = softmax(x)
  #####################################################################
  softmax_dx_local = nl.ndarray((par_dim(q_seq_tile_size), k_seq_tile_size), dtype=kernel_dtype, buffer=nl.sbuf)
  softmax_dx_local[ip_qk, if_qk] = \
    nisa.tensor_scalar(data=softmax_dy[ip_qk, if_qk],
                        op0=np.subtract,
                        operand0=dy_o_sum[local_i_q_seq_tile, ip_qk, 0],
                        op1=np.multiply,
                        operand1=softmax_y[ip_qk, if_qk],
                        mask=mask)

  #####################################################################
  # Step 5.1 Calculate dK, with matmul(LHS=softmax_dx, RHS=Q)
  #####################################################################
  for i_d_head_tile in nl.affine_range(d_head_n_tiles):
    ip_trans_q = nl.arange(d_head_tile_size)[:, None]
    if_trans_q = nl.arange(q_seq_tile_size)[None, :]
    ip_dk = nl.arange(d_head_tile_size)[:, None]
    trans_q_local = nisa.sb_transpose(q_local[i_d_head_tile, ip_trans_q, if_trans_q],
                                      mask=mask)
    dk_psum[i_d_head_tile, ip_dk, if_qk] += \
      nisa.nc_matmul(trans_q_local,
                      softmax_dx_local[ip_qk, if_qk],
                      mask=mask)

  #####################################################################
  # Step 5.2 Calculate dQ
  #####################################################################
  ip_k = nl.arange(d_head_tile_size)[:, None]
  if_k = nl.arange(k_seq_tile_size_backward)[None, :]
  ip_dq = nl.arange(d_head_tile_size)[:, None]
  if_dq = nl.arange(q_seq_tile_size)[None, :]
  if_d = nl.arange(d_head_tile_size)[None, :]
  ip_transposed_k = nl.arange(k_seq_tile_size_backward)[:, None]
  for i_d_head_tile in nl.affine_range(d_head_n_tiles):
    dq_psum = nl.zeros((par_dim(d_head_tile_size), q_seq_tile_size),
                        dtype=np.float32, buffer=nl.psum)
    for i_k_seq_tile_backward in nl.affine_range(k_seq_fwd_bwd_tile_multipler):
      transposed_softmax_dx_local = \
        nisa.sb_transpose(softmax_dx_local[ip_qk, i_k_seq_tile_backward * k_seq_tile_size_backward + if_k],
                          mask=mask)
      dq_psum[ip_dq, if_dq] += nisa.nc_matmul(
          transposed_k_local[i_k_seq_tile_backward, i_d_head_tile, ip_transposed_k, if_d],
          transposed_softmax_dx_local,
          mask=mask)
    dq_local = nl.multiply(dq_psum[ip_dq, if_dq], softmax_scale, dtype=kernel_dtype, mask=mask)
    dq_local_reduced[local_i_q_seq_tile, i_d_head_tile, ip_dq, if_dq] = nl.loop_reduce(
      dq_local, op=np.add, loop_indices=(local_i_k_seq_tile,),
      dtype=mixed_dtype, mask=mask)

def flash_attn_bwd(
  q_ref, k_ref, v_ref, o_ref,
  dy_ref,
  lse_ref,
  seed_ref,
  out_dq_ref, out_dk_ref, out_dv_ref,
  use_causal_mask=False,
  mixed_precision=False,
  dropout_p=0.0,
  softmax_scale=None,
):
  """
  Flash attention backward kernel. Compute the backward gradients.

  IO tensor layouts:
   - q_ref: shape (bs, nheads, head_size, seq)
   - k_ref: shape (bs, nheads, head_size, seq)
   - v_ref: shape (bs, nheads, head_size, seq)
   - o_ref: shape (bs, nheads, head_size, seq)
   - dy_ref: shape (bs, nheads, head_size, seq)
   - lse_ref: shape (bs, nheads, nl.tile_size.pmax, seq // nl.tile_size.pmax)
   - out_dq_ref: shape (bs, nheads, head_size, seq)
   - out_dk_ref: shape (bs, nheads, head_size, seq)
   - out_dv_ref: shape (bs, nheads, head_size, seq)

  Detailed steps:
    1. D = rowsum(dO ◦ O) (pointwise multiply)

    2. Recompute (softmax(Q@K^T))
      2.1 Q@K^T
      2.2 Scale the QK score
      2.3 Apply causal mask
      2.4 softmax
    3. Compute the gradients of y = score @ V with respect to the loss

    4. Compute the gradients of y = softmax(x)

    5. Compute the gradients of Q@K^T
      4.1 Compute dQ
      4.2 Compute dK
  """
  use_causal_mask=True
  mixed_precision=True
  dropout_p=0.0
  # Use q_ref dtype as the intermediate tensor dtype
  # Assume all IO tensors have the same dtype
  kernel_dtype = q_ref.dtype
  mixed_dtype = np.dtype(np.float32) if mixed_precision else kernel_dtype

  assert q_ref.dtype == k_ref.dtype == v_ref.dtype == o_ref.dtype == dy_ref.dtype \
         == out_dq_ref.dtype == out_dk_ref.dtype == out_dv_ref.dtype
  assert lse_ref.dtype == mixed_dtype

  # Shape checking
  bs, nheads, d_head, seqlen = q_ref.shape
  assert tuple(k_ref.shape) == (bs, nheads, d_head, seqlen), \
    f"Input K shape mismatch, got {k_ref.shape}"
  assert tuple(v_ref.shape) == (bs, nheads, d_head, seqlen), \
    f"Input V shape mismatch, got {v_ref.shape}"
  assert tuple(o_ref.shape) == (bs, nheads, d_head, seqlen), \
    f"Input dy shape mismatch, got {o_ref.shape}"
  assert tuple(dy_ref.shape) == (bs, nheads, d_head, seqlen), \
    f"Input dy shape mismatch, got {dy_ref.shape}"
  assert tuple(lse_ref.shape) == (bs, nheads, nl.tile_size.pmax, seqlen // nl.tile_size.pmax), \
    f"Input lse shape mismatch, got {lse_ref.shape}"
  if seed_ref is not None:
    assert tuple(seed_ref.shape) == (1,), \
      f"Input seed shape mismatch, got {seed_ref.shape}"

  assert tuple(out_dq_ref.shape) == (bs, nheads, d_head, seqlen), \
    f"Output dQ shape mismatch, got {out_dq_ref.shape}"
  assert tuple(out_dk_ref.shape) == (bs, nheads, d_head, seqlen), \
    f"Output dK shape mismatch, got {out_dk_ref.shape}"
  assert tuple(out_dv_ref.shape) == (bs, nheads, d_head, seqlen), \
    f"Output dV shape mismatch, got {out_dv_ref.shape}"

  # FIXME: Add masking for different seqlen values.
  assert seqlen % 128 == 0, \
    f"Input sequence length must be divisible by 128, got {seqlen}"

  # Softmax scaling factor, multiplied onto Q
  #softmax_scale = softmax_scale or 1.0 / float(d_head ** 0.5)
  softmax_scale = 1.0 / float(d_head ** 0.5)

  # Different batch samples/attention heads have independent attention
  batch_id = nl.program_id(axis=0)
  head_id = nl.program_id(axis=1)

  q_seq_n_tiles, q_seq_tile_size = div_ceil(seqlen, 128), 128
  d_head_n_tiles, d_head_tile_size = div_ceil(d_head, 128), min(d_head, 128)

  if seqlen >= 512:
    k_seq_n_tiles, k_seq_tile_size = seqlen // 512, 512
  else:
    k_seq_n_tiles, k_seq_tile_size = seqlen // 128, 128

  k_seq_n_tiles_backward, k_seq_tile_size_backward = seqlen // 128, 128
  k_seq_fwd_bwd_tile_multipler = k_seq_tile_size // k_seq_tile_size_backward

  ##############################################################
  # Step 2.4 Prefetch exp bias for softmax
  ##############################################################
  softmax_exp_bias = nl.zeros((q_seq_n_tiles, par_dim(q_seq_tile_size), 1), dtype=mixed_dtype)
  for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):
    ip_qk = nl.arange(q_seq_tile_size)[:, None]
    lse_local = nl.load(
      lse_ref[batch_id, head_id, ip_qk, i_q_seq_tile],
      dtype=mixed_dtype)
    softmax_exp_bias[i_q_seq_tile, ip_qk, 0] = lse_local * -1.0

  ##############################################################
  # Step 1 Compute rowsum(dO ◦ O)
  ##############################################################
  dy_o_sum = nl.ndarray((q_seq_n_tiles, par_dim(q_seq_tile_size), 1), dtype=mixed_dtype)
  for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):
    ip_reduce = nl.arange(q_seq_tile_size)[:, None]
    dy_o_partial = nl.zeros((par_dim(q_seq_tile_size), d_head_n_tiles), dtype=mixed_dtype)
    for i_d_head_tile in nl.affine_range(d_head_n_tiles):
      ip_load = nl.arange(d_head_tile_size)[:, None]
      if_q = nl.arange(q_seq_tile_size)[None, :]
      dy_local = nl.load_transpose2d(
        dy_ref[batch_id, head_id, i_d_head_tile * d_head_tile_size + ip_load, i_q_seq_tile * q_seq_tile_size + if_q],
        dtype=mixed_dtype)
      o_local = nl.load_transpose2d(
        o_ref[batch_id, head_id, i_d_head_tile * d_head_tile_size + ip_load, i_q_seq_tile * q_seq_tile_size + if_q],
        dtype=mixed_dtype
      )

      dy_o_partial[ip_reduce, i_d_head_tile] = nisa.reduce(
        np.add, data=dy_local*o_local, axis=(1,), dtype=mixed_dtype
      )

    dy_o_sum[i_q_seq_tile, ip_reduce, 0] = nisa.reduce(
      np.add, data=dy_o_partial[ip_reduce, nl.arange(d_head_n_tiles)[None, :]],
      axis=(1,), dtype=mixed_dtype
    )

  # Indices for prefetch
  ip_qk = nl.arange(d_head_tile_size)[:, None]
  if_q = nl.arange(q_seq_tile_size)[None, :]
  if_k = nl.arange(k_seq_tile_size)[None, :]

  if dropout_p > 0.0:
    seed_local = nl.ndarray((par_dim(1), 1), buffer=nl.sbuf, dtype=nl.int32)
    seed_local[0, 0] = nl.load(seed_ref[0])
    # TODO: Remove this once the dropout supports scale prob
    dropout_p_local = nl.full((q_seq_tile_size, 1), fill_value=dropout_p, dtype=np.float32)
  else:
    seed_local = None
    dropout_p_local = None

  dq_local_reduced = nl.zeros((q_seq_n_tiles, d_head_n_tiles, par_dim(d_head_tile_size), q_seq_tile_size),
                              dtype=mixed_dtype)

  # affine_range give the compiler permission to vectorize instructions
  # inside the loop which improves the performance. However, when using the 
  # the dropout we should use sequential_range to avoid setting
  # seed vectorization. TODO: the compiler should avoid vectorizing seed setting
  _range = nl.sequential_range if dropout_p > 0.0 else nl.affine_range
  
  for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
    # Prefetch V, K
    v_local = nl.zeros((d_head_n_tiles, par_dim(d_head_tile_size), k_seq_tile_size), dtype=kernel_dtype)
    k_local = nl.zeros((d_head_n_tiles, par_dim(d_head_tile_size), k_seq_tile_size), dtype=kernel_dtype)
    transposed_k_local = nl.zeros((k_seq_fwd_bwd_tile_multipler, d_head_n_tiles, par_dim(k_seq_tile_size_backward), d_head_tile_size), dtype=kernel_dtype)
    for i_d_head_tile in nl.affine_range(d_head_n_tiles):
      k_local[i_d_head_tile, ip_qk, if_k] = nl.load(
        k_ref[batch_id, head_id, i_d_head_tile * d_head_tile_size + ip_qk, i_k_seq_tile * k_seq_tile_size + if_k],
        dtype=kernel_dtype)
      v_local[i_d_head_tile, ip_qk, if_k] = nl.load(
        v_ref[batch_id, head_id, i_d_head_tile * d_head_tile_size + ip_qk, i_k_seq_tile * k_seq_tile_size + if_k],
        dtype=kernel_dtype)
      ##############################################################
      # Prefetch k transpose for the backward too
      ##############################################################
      if_k_backward = nl.arange(k_seq_tile_size_backward)[None, :]
      ip_k_backward = nl.arange(k_seq_tile_size_backward)[:, None]
      if_d_head = nl.arange(d_head_tile_size)[None, :]
      for i_k_seq_tile_backward in nl.affine_range(k_seq_fwd_bwd_tile_multipler):
        transposed_k_local[i_k_seq_tile_backward, i_d_head_tile, ip_k_backward, if_d_head] = \
          nisa.sb_transpose(k_local[i_d_head_tile, ip_qk,
                                    i_k_seq_tile_backward * k_seq_tile_size_backward + if_k_backward])

    dv_psum = nl.zeros((d_head_n_tiles, par_dim(d_head_tile_size), k_seq_tile_size),
                        dtype=np.float32, buffer=nl.psum)
    dk_psum = nl.zeros((d_head_n_tiles, par_dim(d_head_tile_size), k_seq_tile_size),
                        dtype=np.float32, buffer=nl.psum)
    for i_q_seq_tile in _range(q_seq_n_tiles):
      # Prefetch dy, Q
      dy_local = nl.zeros((d_head_n_tiles, par_dim(d_head_tile_size), q_seq_tile_size), dtype=kernel_dtype)
      q_local = nl.zeros((d_head_n_tiles, par_dim(d_head_tile_size), q_seq_tile_size), dtype=kernel_dtype)
      for i_d_head_tile in nl.affine_range(d_head_n_tiles):
        ip_qk = nl.arange(d_head_tile_size)[:, None]
        if_q = nl.arange(q_seq_tile_size)[None, :]

        dy_local[i_d_head_tile, ip_qk, if_q] = nl.load(
          dy_ref[batch_id, head_id, i_d_head_tile * d_head_tile_size + ip_qk, i_q_seq_tile * q_seq_tile_size + if_q],
          dtype=kernel_dtype)

        q_local[i_d_head_tile, ip_qk, if_q] = nl.load(
          q_ref[batch_id, head_id, i_d_head_tile * d_head_tile_size + ip_qk, i_q_seq_tile * q_seq_tile_size + if_q],
          dtype=kernel_dtype) * softmax_scale

      _flash_attn_bwd_core(
        q_local=q_local, k_local=k_local, transposed_k_local=transposed_k_local,
        v_local=v_local, dy_local=dy_local,
        dk_psum=dk_psum, dv_psum=dv_psum, dq_local_reduced=dq_local_reduced,
        softmax_exp_bias=softmax_exp_bias, dy_o_sum=dy_o_sum,
        local_i_q_seq_tile=i_q_seq_tile, local_i_k_seq_tile=i_k_seq_tile,
        seqlen=seqlen, d_head=d_head, 
        use_causal_mask=use_causal_mask,
        kernel_dtype=kernel_dtype, mixed_dtype=mixed_dtype,
        softmax_scale=softmax_scale,
        seed_local=seed_local, dropout_p=dropout_p, dropout_p_local=dropout_p_local,       
      )
     
    # Write dK, dV
    for i_d_head_tile in nl.affine_range(d_head_n_tiles):
      ip_dkv = nl.arange(d_head_tile_size)[:, None]
      if_dkv = nl.arange(k_seq_tile_size)[None, :]

      nl.store(
        out_dv_ref[batch_id, head_id,
                   i_d_head_tile * d_head_tile_size + ip_dkv,
                   i_k_seq_tile * k_seq_tile_size + if_dkv],
        value=dv_psum[i_d_head_tile, ip_dkv, if_dkv],
      )

      nl.store(
        out_dk_ref[batch_id, head_id,
                    i_d_head_tile * d_head_tile_size + ip_dkv,
                    i_k_seq_tile * k_seq_tile_size + if_dkv],
        value=dk_psum[i_d_head_tile, ip_dkv, if_dkv],
      )

  # Write dQ
  for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):
    for i_d_head_tile in nl.affine_range(d_head_n_tiles):
      ip_dq = nl.arange(d_head_tile_size)[:, None]
      if_dq = nl.arange(q_seq_tile_size)[None, :]

      nl.store(
        out_dq_ref[batch_id, head_id,
                   i_d_head_tile * d_head_tile_size + ip_dq,
                   i_q_seq_tile * q_seq_tile_size + if_dq],
        value=dq_local_reduced[i_q_seq_tile, i_d_head_tile, ip_dq, if_dq],
      )


def fused_self_attn_for_SD(q_ref, k_ref, v_ref, out_ref, use_causal_mask=False,
                           mixed_percision=True):
  """
  Fused self attention kernel. Computes softmax(QK^T)V. Decoder model
  can optionally include a causal mask application. Does not include QKV
  projection, output projection, dropout, residual connection, etc.

  IO tensor layouts:
  -- Intended to be the same to BIR kernel
   - q_ptr: shape   (bs, head_size, seq_q)
   - k_ptr: shape   (bs, seq_k, head_size)
   - v_ptr: shape   (bs, seq_v, head_size)
   - out_ptr: shape (bs, seq_q, head_size)
   - We use seq_q and seq_k just for clarity, this kernel requires seq_q == seq_k

  IO tensor dtypes:
   - This kernel assumes all IO tensors have the same dtype
   - Intermediate tensor dtypes will use the same dtype as IO tensors
  """
  # Use q_ref dtype as the intermediate tensor dtype
  # Assume all IO tensors have the same dtype
  kernel_dtype = q_ref.dtype
  pe_in_dt = nl.bfloat16 if mixed_percision else np.float32
  assert q_ref.dtype == k_ref.dtype == v_ref.dtype == out_ref.dtype

  # Shape checking
  bs, d_head, seqlen = q_ref.shape
  assert tuple(q_ref.shape) == (bs, d_head, seqlen), 'Input shape mismatch!'
  assert tuple(k_ref.shape) == (bs, seqlen, d_head), 'Input shape mismatch!'
  assert tuple(v_ref.shape) == (bs, seqlen,
                                d_head), f'Input shape mismatch! Expected: {(bs, seqlen, d_head)} Actual: {tuple(v_ref.shape)}'
  assert tuple(out_ref.shape) == (bs, seqlen, d_head), 'Output shape mismatch!'

  # Softmax scaling factor, multiplied onto Q
  softmax_scale = 0.125

  # Different batch samples/attention heads have independent attention
  # FIXME: use program_id errors out in inline procedure
  # batch_id = nl.program_id(axis=0)

  # TODO: make q_seq_tile_size user input
  # The matmuls currently use a fixed tile size of (128, 128). This may not achieve the best
  # performance for dense attention. However, since this kernel is in preparation
  # for block-sparse attention, this tile size is acceptable because the block
  # size of block-sparse attention cannot be too large.
  q_seq_n_tiles, q_seq_tile_size = seqlen // 128, 128
  k_seq_n_tiles, k_seq_tile_size = seqlen // 128, 128
  d_head_n_tiles, d_head_tile_size = d_head // 128, 128
  v_seq_n_tiles, v_seq_tile_size = seqlen // 128, 128

  for batch_id in nl.affine_range(bs):
    ###################################
    # Step 1. transpose(tensor_v)
    ###################################
    # Buffer for v matrix transposed
    # Pre-fetch and keep it in SBUF throughout different softmax tiles
    trans_v = nl.ndarray((par_dim(v_seq_tile_size), v_seq_n_tiles, d_head), dtype=pe_in_dt)

    for i_d_head_tile in nl.affine_range(d_head_n_tiles):
      for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
        ip_v = nl.arange(v_seq_tile_size)[:, None]
        if_v = nl.arange(d_head_tile_size)[None, :]
        trans_v[ip_v, i_k_seq_tile, i_d_head_tile * d_head_tile_size + if_v] = nl.load(
          v_ref[
            batch_id, i_k_seq_tile * k_seq_tile_size + ip_v, i_d_head_tile * d_head_tile_size + if_v],
          dtype=pe_in_dt)

    q_local = nl.ndarray(
      (q_seq_n_tiles, d_head_n_tiles, par_dim(d_head_tile_size), q_seq_tile_size), dtype=pe_in_dt)
    ip_q = nl.arange(d_head_tile_size)[:, None]
    if_q = nl.arange(q_seq_tile_size)[None, :]
    for i_d_head_tile in nl.affine_range(d_head_n_tiles):
      for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):
        q_local[i_q_seq_tile, i_d_head_tile, ip_q, if_q] = nl.load(
          q_ref[
            batch_id, i_d_head_tile * d_head_tile_size + ip_q, i_q_seq_tile * q_seq_tile_size + if_q],
          dtype=pe_in_dt) * softmax_scale

    k_local = nl.ndarray(
      (k_seq_n_tiles, d_head_n_tiles, par_dim(d_head_tile_size), k_seq_tile_size), dtype=pe_in_dt)
    ip_k = nl.arange(d_head_tile_size)[:, None]
    if_k = nl.arange(k_seq_tile_size)[None, :]
    for i_d_head_tile in nl.affine_range(d_head_n_tiles):
      for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
        k_local[i_k_seq_tile, i_d_head_tile, ip_k, if_k] = nl.load_transpose2d(
          k_ref[
            batch_id,
            i_k_seq_tile * k_seq_tile_size + nl.arange(k_seq_tile_size)[:, None],
            i_d_head_tile * d_head_tile_size + nl.arange(d_head_tile_size)[None, :]],
          dtype=pe_in_dt)

    for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):  # indent = 2
      # A SBUF buffer for an independent softmax tile
      qk_res_buf = nl.ndarray((par_dim(q_seq_tile_size), seqlen), dtype=kernel_dtype)

      neg_max_res = nl.ndarray((par_dim(q_seq_tile_size), k_seq_n_tiles), dtype=kernel_dtype)
      ip_max = nl.arange(q_seq_tile_size)[:, None]
      if_max = nl.arange(k_seq_n_tiles)[None, :]

      # Loop over LHS free of matmul(lhs=tensor_k, rhs=tensor_q, contract=d_head)
      for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):  # indent = 4

        # Since the K^T tile is the LHS, the q_seq_len dimension will be P in the result
        # PSUM buffer shape: [q_seq_tile_size P, k_seq_tile_size F]
        qk_psum = nl.zeros((par_dim(q_seq_tile_size), k_seq_tile_size),
                           dtype=np.float32, buffer=nl.psum)

        # Tensor indices for accessing qk result in k_seq_tile_size
        ip_qk = nl.arange(q_seq_tile_size)[:, None]
        if_qk = nl.arange(k_seq_tile_size)[None, :]

        # Loop over contraction dim of Step 1 matmul
        for i_d_head_tile in nl.affine_range(d_head_n_tiles):  # indent = 6
          ##############################################################
          # Step 2. matmul(lhs=tensor_k, rhs=tensor_q, contract=d_head)
          ##############################################################
          qk_psum[ip_qk, if_qk] += nisa.nc_matmul(rhs=k_local[i_k_seq_tile, i_d_head_tile, ip_k, if_k],
                                                  lhs=q_local[i_q_seq_tile, i_d_head_tile, ip_q, if_q])

          ###################################
          # Step 3. Apply optional causal mask
          ###################################
          if use_causal_mask:
            # Magic number -9984.0 to replace -inf similar to what Tensorizer uses
            qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk] = nisa.affine_select(
              pred=(
                    i_q_seq_tile * q_seq_tile_size + ip_qk >= i_k_seq_tile * k_seq_tile_size + if_qk),
              on_true_tile=qk_psum[ip_qk, if_qk], on_false_value=-9984.0, dtype=kernel_dtype)
          else:
            # Simply send psum result back to sbuf
            qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk] = nl.copy(
              qk_psum[ip_qk, if_qk],
              dtype=kernel_dtype)

        ###################################
        # Step 4. Softmax
        ###################################
        # TODO: use TensorScalarCacheReduce to avoid an extra copy
        # We want to break this reduction in tiles because we want to overlap it with the previous matmul
        neg_max_res[ip_max, i_k_seq_tile] = nisa.reduce(
          np.max, data=qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk],
          axis=(1,), dtype=kernel_dtype, negate=True)

      neg_max_res_final = nisa.reduce(
        np.min, data=neg_max_res[ip_max, if_max],
        axis=(1,), dtype=kernel_dtype, negate=False)

      ip_softmax = nl.arange(q_seq_tile_size)[:, None]
      if_softmax = nl.arange(seqlen)[None, :]
      ip_sum_res = nl.arange(q_seq_tile_size)[:, None]
      if_sum_res = nl.arange(d_head_tile_size)[None, :]

      softmax_res = nl.ndarray((par_dim(q_seq_tile_size), seqlen), dtype=pe_in_dt)
      sum_divisor = nl.ndarray((par_dim(q_seq_tile_size), d_head_tile_size), dtype=kernel_dtype)

      # Simply use a large tile of seq_len in size since this is a "blocking" instruction
      # Assuming the compiler will merge exp and reduce_add into a single instruction on ACT
      exp_res = nisa.activation(np.exp,
                                data=qk_res_buf[ip_softmax, if_softmax],
                                bias=neg_max_res_final, scale=1.0)

      sum_res = nisa.reduce(np.add, data=exp_res, axis=(1,),
                            dtype=kernel_dtype)
      softmax_res[ip_softmax, if_softmax] = nl.copy(exp_res, dtype=pe_in_dt)

      sum_reciprocal_broadcast = (1.0 / sum_res).broadcast_to((q_seq_tile_size, d_head_tile_size))
      sum_divisor[ip_sum_res, if_sum_res] = nl.copy(sum_reciprocal_broadcast, dtype=kernel_dtype)

      # Loop over matmul_1 RHS free
      for i_d_head_tile in nl.affine_range(d_head_n_tiles):

        # Buffer for transposed softmax results (FP32 in PSUM)
        trans_softmax_res = nl.ndarray(
          (par_dim(k_seq_tile_size), k_seq_n_tiles, q_seq_tile_size),
          dtype=pe_in_dt)

        # Result psum buffer has the hidden dim as P
        attn_res_psum = nl.zeros((par_dim(d_head_tile_size), q_seq_tile_size),
                                 dtype=np.float32, buffer=nl.psum)

        ip_scores_t = nl.arange(k_seq_tile_size)[:, None]
        if_scores_t = nl.arange(q_seq_tile_size)[None, :]
        # Loop over matmul_1 contraction
        for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
          ###################################
          # Step 5. transpose(softmax_res)
          ###################################
          ip_scores = nl.arange(q_seq_tile_size)[:, None]
          if_scores = nl.arange(k_seq_tile_size)[None, :]

          trans_softmax_res[ip_scores_t, i_k_seq_tile, if_scores_t] = nisa.sb_transpose(
            softmax_res[ip_scores, i_k_seq_tile * k_seq_tile_size + if_scores])

        ip_out = nl.arange(d_head_tile_size)[:, None]
        if_out = nl.arange(q_seq_tile_size)[None, :]
        for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
          ######################################################################
          # Step 6. matmul_1(lhs=trans_softmax_res, rhs=trans_v, contract=seqlen_v=seqlen_k)
          ######################################################################
          ip_v_t = nl.arange(k_seq_tile_size)[:, None]
          if_v_t = nl.arange(d_head_tile_size)[None, :]
          attn_res_psum[ip_out, if_out] += \
            nisa.nc_matmul(rhs=trans_softmax_res[ip_scores_t, i_k_seq_tile, if_scores_t],
                           lhs=trans_v[ip_v_t, i_k_seq_tile, i_d_head_tile * d_head_tile_size + if_v_t])

        attn_res_sbuf = nl.copy(attn_res_psum[ip_out, if_out], dtype=kernel_dtype)

        attn_res_div = attn_res_sbuf * nisa.sb_transpose(sum_divisor[ip_sum_res, if_sum_res])

        nl.store(
          out_ref[
            batch_id, i_q_seq_tile * q_seq_tile_size + if_out, i_d_head_tile * d_head_tile_size + ip_out],
          value=attn_res_div)


def fused_self_attn_for_SD_small_head_size(q_ref, k_ref, v_ref, out_ref, use_causal_mask=False,
                                           mixed_percision=True):
  """
  Fused self attention kernel for small head size Stable Diffusion workload. 
  
  Computes softmax(QK^T)V. Decoder model can optionally include a causal mask 
  application. Does not include QKV rojection, output projection, dropout, 
  residual connection, etc.

  This kernel is designed to be used for Stable Diffusion models where the 
  n_heads is smaller or equal to 128. Assertion is thrown if `n_heads` does
  not satisfy the requirement.

  IO tensor layouts:
   - q_ptr: shape   (bs, n_heads, seq_q)
   - k_ptr: shape   (bs, seq_k, n_heads)
   - v_ptr: shape   (bs, seq_v, n_heads)
   - out_ptr: shape (bs, seq_q, n_heads)
   - We use seq_q and seq_k just for clarity, this kernel requires seq_q == seq_k

  IO tensor dtypes:
   - This kernel assumes all IO tensors have the same dtype
   - If mixed_percision is True, then all Tensor Engine operation will be performed in
   bfloat16 and accumulation will be performed in float32. Otherwise the intermediates
   will be in the same type as the inputs.
  """
  # Use q_ref dtype as the intermediate tensor dtype
  # Assume all IO tensors have the same dtype
  kernel_dtype = q_ref.dtype
  pe_in_dt = nl.bfloat16 if mixed_percision else np.float32
  assert q_ref.dtype == k_ref.dtype == v_ref.dtype == out_ref.dtype

  # Shape checking
  bs, d_head, seqlen = q_ref.shape
  assert d_head <= 128, "Cannot use this kernel for d_head > 128"
  assert tuple(q_ref.shape) == (bs, d_head, seqlen), 'Input shape mismatch!'
  assert tuple(k_ref.shape) == (bs, seqlen, d_head), 'Input shape mismatch!'
  assert tuple(v_ref.shape) == (bs, seqlen,
                                d_head), f'Input shape mismatch! Expected: {(bs, seqlen, d_head)} Actual: {tuple(v_ref.shape)}'
  assert tuple(out_ref.shape) == (bs, seqlen, d_head), 'Output shape mismatch!'

  # Softmax scaling factor, multiplied onto Q
  softmax_scale = 0.125

  # Different batch samples/attention heads have independent attention
  batch_id = nl.program_id(axis=0)
  # batch_id = 0

  # TODO: make q_seq_tile_size user input
  # The matmuls currently use a fixed tile size of (128, 128). This may not achieve the best
  # performance for dense attention. However, since this kernel is in preparation
  # for block-sparse attention, this tile size is acceptable because the block
  # size of block-sparse attention cannot be too large.
  q_seq_n_tiles, q_seq_tile_size = seqlen // 128, 128
  k_seq_n_tiles, k_seq_tile_size = seqlen // 128, 128
  # No tiling on d_head dimension since the number of d_head fits in SB
  d_head_tile_size = d_head
  v_seq_n_tiles, v_seq_tile_size = seqlen // 128, 128

  ###################################
  # Step 1. transpose(tensor_v)
  ###################################
  # Buffer for v matrix transposed
  # Pre-fetch and keep it in SBUF throughout different softmax tiles
  trans_v = nl.ndarray((par_dim(v_seq_tile_size), v_seq_n_tiles, d_head), dtype=pe_in_dt)

  for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
    ip_v = nl.arange(v_seq_tile_size)[:, None]
    if_v = nl.arange(d_head_tile_size)[None, :]
    trans_v[ip_v, i_k_seq_tile, if_v] = nl.load(
      v_ref[batch_id, i_k_seq_tile * k_seq_tile_size + ip_v, if_v],
      dtype=pe_in_dt)

  q_local = nl.ndarray((q_seq_n_tiles, par_dim(d_head_tile_size), q_seq_tile_size), dtype=pe_in_dt)
  ip_q = nl.arange(d_head_tile_size)[:, None]
  if_q = nl.arange(q_seq_tile_size)[None, :]
  for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):
    q_local[i_q_seq_tile, ip_q, if_q] = nl.load(
      q_ref[batch_id, ip_q, i_q_seq_tile * q_seq_tile_size + if_q],
      dtype=pe_in_dt) * softmax_scale

  k_local = nl.ndarray((k_seq_n_tiles, par_dim(d_head_tile_size), k_seq_tile_size), dtype=pe_in_dt)
  ip_k = nl.arange(d_head_tile_size)[:, None]
  if_k = nl.arange(k_seq_tile_size)[None, :]
  for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
    k_local[i_k_seq_tile, ip_k, if_k] = nl.load_transpose2d(
      k_ref[batch_id,
            i_k_seq_tile * k_seq_tile_size + nl.arange(k_seq_tile_size)[:, None],
            nl.arange(d_head_tile_size)[None, :]],
      dtype=pe_in_dt)

  for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):  # indent = 2
    # A SBUF buffer for an independent softmax tile
    qk_res_buf = nl.ndarray((par_dim(q_seq_tile_size), seqlen), dtype=kernel_dtype)

    neg_max_res = nl.ndarray((par_dim(q_seq_tile_size), k_seq_n_tiles), dtype=kernel_dtype)
    ip_max = nl.arange(q_seq_tile_size)[:, None]
    if_max = nl.arange(k_seq_n_tiles)[None, :]

    # Loop over LHS free of matmul(lhs=tensor_k, rhs=tensor_q, contract=d_head)
    for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):  # indent = 4

      # Since the K^T tile is the LHS, the q_seq_len dimension will be P in the result
      # PSUM buffer shape: [q_seq_tile_size P, k_seq_tile_size F]
      qk_psum = nl.zeros((par_dim(q_seq_tile_size), k_seq_tile_size),
                         dtype=np.float32, buffer=nl.psum)

      # Tensor indices for accessing qk result in k_seq_tile_size
      ip_qk = nl.arange(q_seq_tile_size)[:, None]
      if_qk = nl.arange(k_seq_tile_size)[None, :]

      ##############################################################
      # Step 2. matmul(lhs=tensor_k, rhs=tensor_q, contract=d_head)
      ##############################################################
      qk_psum[ip_qk, if_qk] += nisa.nc_matmul(rhs=k_local[i_k_seq_tile, ip_k, if_k],
                                              lhs=q_local[i_q_seq_tile, ip_q, if_q])

      ###################################
      # Step 3. Apply optional causal mask
      ###################################
      if use_causal_mask:
        # Magic number -9984.0 to replace -inf similar to what Tensorizer uses
        qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk] = nisa.affine_select(
          pred=(i_q_seq_tile * q_seq_tile_size + ip_qk >= i_k_seq_tile * k_seq_tile_size + if_qk),
          on_true_tile=qk_psum[ip_qk, if_qk], on_false_value=-9984.0, dtype=kernel_dtype)
      else:
        # Simply send psum result back to sbuf
        qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk] = nl.copy(qk_psum[ip_qk, if_qk],
                                                                              dtype=kernel_dtype)

      ###################################
      # Step 4. Softmax
      ###################################
      # TODO: use TensorScalarCacheReduce to avoid an extra copy
      # We want to break this reduction in tiles because we want to overlap it with the previous matmul
      neg_max_res[ip_max, i_k_seq_tile] = nisa.reduce(
        np.max, data=qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk],
        axis=(1,), dtype=kernel_dtype, negate=True)

    neg_max_res_final = nisa.reduce(
      np.min, data=neg_max_res[ip_max, if_max],
      axis=(1,), dtype=kernel_dtype, negate=False)

    ip_softmax = nl.arange(q_seq_tile_size)[:, None]
    if_softmax = nl.arange(seqlen)[None, :]
    ip_sum_res = nl.arange(q_seq_tile_size)[:, None]
    if_sum_res = nl.arange(d_head_tile_size)[None, :]

    softmax_res = nl.ndarray((par_dim(q_seq_tile_size), seqlen), dtype=pe_in_dt)
    sum_divisor = nl.ndarray((par_dim(q_seq_tile_size), d_head_tile_size), dtype=kernel_dtype)

    # Simply use a large tile of seq_len in size since this is a "blocking" instruction
    # Assuming the compiler will merge exp and reduce_add into a single instruction on ACT
    exp_res = nisa.activation(np.exp,
                              data=qk_res_buf[ip_softmax, if_softmax],
                              bias=neg_max_res_final, scale=1.0)

    sum_res = nisa.reduce(np.add, data=exp_res, axis=(1,),
                          dtype=kernel_dtype)
    softmax_res[ip_softmax, if_softmax] = nl.copy(exp_res, dtype=pe_in_dt)

    sum_reciprocal_broadcast = (1.0 / sum_res).broadcast_to((q_seq_tile_size, d_head_tile_size))
    sum_divisor[ip_sum_res, if_sum_res] = nl.copy(sum_reciprocal_broadcast, dtype=kernel_dtype)

    # Buffer for transposed softmax results (FP32 in PSUM)
    trans_softmax_res = nl.ndarray(
      (par_dim(k_seq_tile_size), k_seq_n_tiles, q_seq_tile_size),
      dtype=pe_in_dt)

    # Result psum buffer has the hidden dim as P
    attn_res_psum = nl.zeros((par_dim(d_head_tile_size), q_seq_tile_size),
                             dtype=np.float32, buffer=nl.psum)

    ip_scores_t = nl.arange(k_seq_tile_size)[:, None]
    if_scores_t = nl.arange(q_seq_tile_size)[None, :]
    # Loop over matmul_1 contraction
    for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
      ###################################
      # Step 5. transpose(softmax_res)
      ###################################
      ip_scores = nl.arange(q_seq_tile_size)[:, None]
      if_scores = nl.arange(k_seq_tile_size)[None, :]

      trans_softmax_res[ip_scores_t, i_k_seq_tile, if_scores_t] = nisa.sb_transpose(
        softmax_res[ip_scores, i_k_seq_tile * k_seq_tile_size + if_scores])

    ip_out = nl.arange(d_head_tile_size)[:, None]
    if_out = nl.arange(q_seq_tile_size)[None, :]
    for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
      ######################################################################
      # Step 6. matmul_1(lhs=trans_softmax_res, rhs=trans_v, contract=seqlen_v=seqlen_k)
      ######################################################################
      ip_v_t = nl.arange(k_seq_tile_size)[:, None]
      if_v_t = nl.arange(d_head_tile_size)[None, :]
      attn_res_psum[ip_out, if_out] += \
        nisa.nc_matmul(rhs=trans_softmax_res[ip_scores_t, i_k_seq_tile, if_scores_t],
                       lhs=trans_v[ip_v_t, i_k_seq_tile, if_v_t])

    attn_res_sbuf = nl.copy(attn_res_psum[ip_out, if_out], dtype=kernel_dtype)

    attn_res_div = attn_res_sbuf * nisa.sb_transpose(sum_divisor[ip_sum_res, if_sum_res])

    nl.store(
      out_ref[batch_id, i_q_seq_tile * q_seq_tile_size + if_out, ip_out],
      value=attn_res_div)

def fused_self_attn_for_SD_small_head_size_asymetric(q_ptr, k_ptr, v_ptr, softmax_scale, out_ptr, mixed_percision=True):
  """
  Fused self attention kernel. Computes softmax(QK^T)V.
  Q_seqlen != K_V_seqlen, hence asymetric

  IO tensor layouts:
  -- Intended to be the same to BIR kernel
   - q_ptr: shape   (bs, head_size, seq_q)
   - k_ptr: shape   (bs, head_size, seq_k_v)
   - v_ptr: shape   (bs, head_size, seq_k_v)
   - out_ptr: shape (bs, head_size, seq_q)

  IO tensor dtypes:
   - This kernel assumes all IO tensors have the same dtype
   - Intermediate tensor dtypes will use the same dtype as IO tensors
  """
  # breakpoint()
  q_ref = q_ptr
  k_ref = k_ptr
  v_ref = v_ptr
  out_ref = out_ptr

  # Use q_ref dtype as the intermediate tensor dtype
  # Assume all IO tensors have the same dtype
  kernel_dtype = q_ref.dtype
  pe_in_dt = nl.bfloat16 if mixed_percision else np.float32
  assert q_ref.dtype == k_ref.dtype == v_ref.dtype == out_ref.dtype

  # Shape checking
  bs, q_d_head, q_seqlen = tuple(q_ref.shape)
  bs, k_d_head, k_seqlen = tuple(k_ref.shape)
  bs, v_d_head, v_seqlen = tuple(v_ref.shape)

  assert q_d_head == k_d_head and k_d_head == v_d_head, "All d_head must be the same"
  assert k_seqlen == v_seqlen, "k_seqlen and k_seqlen must be the same"

  d_head = q_d_head

  assert d_head <= 128, "Cannot use this kernel for d_head > 128"
  assert k_seqlen <= 128, "Cannot use this kernel for k_seqlen > 128"
  assert v_seqlen <= 128, "Cannot use this kernel for v_seqlen > 128"
  assert q_seqlen % 128 == 0, "q_seqlen must be multiple for 128"

  batch_id = nl.program_id(axis=0)

  # Tiling sizes
  q_seq_n_tiles, q_seq_tile_size = q_seqlen // 128, 128
  k_seq_n_tiles, k_seq_tile_size = 1, k_seqlen
  d_head_tile_size = d_head
  v_seq_n_tiles, v_seq_tile_size = 1, v_seqlen

  ##############################################################
  # Step 1. Load input tensors
  ##############################################################

  # Load transposed value tensor
  trans_v = nl.ndarray((par_dim(v_seq_tile_size), d_head), dtype=pe_in_dt)
  ip_v = nl.arange(v_seq_tile_size)[:, None]
  if_v = nl.arange(d_head_tile_size)[None, :]
  trans_v[ip_v, if_v] = nl.load_transpose2d(
    v_ref[batch_id, nl.arange(d_head_tile_size)[:, None], nl.arange(v_seq_tile_size)[None, :]],
    dtype=pe_in_dt)

  # Load query tensor
  q_local = nl.ndarray((q_seq_n_tiles, par_dim(d_head_tile_size), q_seq_tile_size), dtype=pe_in_dt)
  ip_q = nl.arange(d_head_tile_size)[:, None]
  if_q = nl.arange(q_seq_tile_size)[None, :]
  for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):
    q_local[i_q_seq_tile, ip_q, if_q] = nl.load(
      q_ref[batch_id, ip_q, i_q_seq_tile * q_seq_tile_size + if_q],
      dtype=pe_in_dt) * softmax_scale

  # Load key tensor
  k_local = nl.ndarray((par_dim(d_head_tile_size), k_seq_tile_size), dtype=pe_in_dt)
  ip_k = nl.arange(d_head_tile_size)[:, None]
  if_k = nl.arange(k_seq_tile_size)[None, :]
  k_local[ip_k, if_k] = nl.load(
    k_ref[batch_id, ip_k, if_k],
    dtype=pe_in_dt)

  for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):  # indent = 2
    # A SBUF buffer for an independent softmax tile
    qk_res_buf = nl.ndarray((par_dim(q_seq_tile_size), k_seq_tile_size), dtype=kernel_dtype)

    qk_psum = nl.zeros((par_dim(q_seq_tile_size), k_seq_tile_size),
                      dtype=np.float32, buffer=nl.psum)

    ip_qk = nl.arange(q_seq_tile_size)[:, None]
    if_qk = nl.arange(k_seq_tile_size)[None, :]

    ##############################################################
    # Step 2. first matmult (lhs=tensor_k, rhs=tensor_q, contract=d_head)
    ##############################################################
    qk_psum[ip_qk, if_qk] += nisa.nc_matmul(rhs=k_local[ip_k, if_k], lhs=q_local[i_q_seq_tile, ip_q, if_q])
    qk_res_buf[ip_qk, if_qk] = nl.copy(qk_psum[ip_qk, if_qk],
                                                                                  dtype=kernel_dtype)

    ###################################
    # Step 3. Softmax
    ###################################
    # Compute max
    neg_max_res = nl.ndarray((par_dim(q_seq_tile_size), k_seq_n_tiles), dtype=kernel_dtype)
    if_neg = nl.arange(k_seq_n_tiles)[None, :]
    neg_max_res[ip_qk, if_neg] = nisa.reduce(
      np.max, data=qk_res_buf[ip_qk, if_qk],
      axis=(1,), dtype=kernel_dtype, negate=True)

    neg_max_res_final = nisa.reduce(
      np.min, data=neg_max_res[ip_qk, if_neg],
      axis=(1,), dtype=kernel_dtype, negate=False)

    # Compute exp
    ip_softmax = nl.arange(q_seq_tile_size)[:, None]
    if_softmax = nl.arange(k_seq_tile_size)[None, :]
    softmax_res = nl.ndarray((par_dim(q_seq_tile_size), k_seq_tile_size), dtype=pe_in_dt)
    exp_res = nisa.activation(np.exp,
                                    data=qk_res_buf[ip_softmax, if_softmax],
                                    bias=neg_max_res_final, scale=1.0)

    # Compute sum of exp
    sum_res = nisa.reduce(np.add, data=exp_res, axis=(1,),
                                    dtype=kernel_dtype)
    sum_reciprocal = 1.0 / sum_res

    # Final softmax result by dividing exp elements by exp sum
    # (multiply by reciprocal is same as division)
    softmax_res[ip_softmax, if_softmax] = exp_res * sum_reciprocal

    trans_softmax_res = nl.ndarray(
      (par_dim(k_seq_tile_size), q_seq_tile_size),
      dtype=pe_in_dt)

    attn_res_psum = nl.zeros((par_dim(d_head_tile_size), q_seq_tile_size),
                            dtype=np.float32, buffer=nl.psum)

    ip_scores_t = nl.arange(k_seq_tile_size)[:, None]
    if_scores_t = nl.arange(q_seq_tile_size)[None, :]

    ###################################
    # Step 4. transpose(softmax_res)
    ###################################
    ip_scores = nl.arange(q_seq_tile_size)[:, None]
    if_scores = nl.arange(k_seq_tile_size)[None, :]

    trans_softmax_res[ip_scores_t, if_scores_t] = nisa.sb_transpose(
      softmax_res[ip_scores, if_scores])

    ip_out = nl.arange(d_head_tile_size)[:, None]
    if_out = nl.arange(q_seq_tile_size)[None, :]

    ######################################################################
    # Step 5. second matmult (lhs=trans_softmax_res, rhs=trans_v, contract=seqlen_v=seqlen_k)
    ######################################################################
    ip_v_t = nl.arange(v_seq_tile_size)[:, None]
    if_v_t = nl.arange(d_head_tile_size)[None, :]
    attn_res_psum[ip_out, if_out] += \
      nisa.nc_matmul(rhs=trans_softmax_res[ip_scores_t, if_scores_t],
                        lhs=trans_v[ip_v_t, if_v_t])

    ######################################################################
    # Step 6. Store result
    ######################################################################
    attn_res_sbuf = nl.ndarray(
      (par_dim(d_head_tile_size), q_seq_tile_size),
      dtype=pe_in_dt)
    attn_res_sbuf[ip_out, if_out] = nl.copy(attn_res_psum[ip_out, if_out], dtype=kernel_dtype)

    nl.store(
      out_ref[batch_id, ip_out, i_q_seq_tile * q_seq_tile_size + if_out],
      value=attn_res_sbuf[ip_out, if_out])


def mamba_prefix_scan_kernel(a, b, out_a, out_b, tile_size):
  def scan_op(x1_0, x1_1, x2_0, x2_1):
    return x1_0 * x2_0, x2_0 * x1_1 + x2_1

  bs, p, n = a.shape
  # FIXME: maximum number of parallel scans is 128 (to fit partition dimension of the architecture)
  assert p <= 128
  assert n % tile_size == 0
  n_tiles = n // tile_size

  n_steps = int(np.floor(np.log2(n)))
  # We create intermediate x for each time step, this avoid loop carried dependence
  xs0 = [nl.ndarray(shape=[p, n], dtype=a.dtype) for _ in range(n_steps + 1)]
  xs1 = [nl.ndarray(shape=[p, n], dtype=a.dtype) for _ in range(n_steps + 1)]

  i_p = nl.arange(p)[:, None]
  i_f = nl.arange(tile_size)[None, :]

  pid_0 = nl.program_id(0)

  for j0 in nl.affine_range(n_tiles):
    j = j0 * tile_size + i_f
    xs0[0][i_p, j] = nl.load(a[pid_0, i_p, j])
    xs1[0][i_p, j] = nl.load(b[pid_0, i_p, j])

  for i in nl.static_range(n_steps):
    for j0 in nl.affine_range(n_tiles):
      j = j0 * tile_size + i_f
      # The following code implements the where operator, without oob access
      # xs_prev0 = nl.where(j - 2 ** i >= 0, xs0[i][j - 2 ** i], 1)
      # xs_prev1 = nl.where(j - 2 ** i >= 0, xs1[i][j - 2 ** i], 0)

      xs_prev0 = nl.ndarray((p, tile_size), dtype=a.dtype)
      xs_prev1 = nl.ndarray((p, tile_size), dtype=a.dtype)

      xs_prev0[:, :] = 1
      xs_prev1[:, :] = 0

      xs_prev0[:, :] = nl.copy(xs0[i][i_p, j - 2 ** i],
                               mask=j >= 2 ** i)
      xs_prev1[:, :] = nl.copy(xs1[i][i_p, j - 2 ** i],
                               mask=j >= 2 ** i)

      xs0[i + 1][i_p, j], xs1[i + 1][i_p, j] = scan_op(xs_prev0, xs_prev1,
                                                       xs0[i][i_p, j],
                                                       xs1[i][i_p, j])

  for j0 in nl.affine_range(n_tiles):
    j = j0 * tile_size + i_f
    nl.store(out_a[pid_0, i_p, j], xs0[-1][i_p, j])
    nl.store(out_b[pid_0, i_p, j], xs1[-1][i_p, j])


@dataclass(frozen=True)
class FlashConfig:
  """
    Config class for flash attention with default values
  """
  seq_tile_size:int = 2048
  training:bool = True
  should_transpose_v:bool = False

  __annotations__ = {
    'seq_tile_size': int,
    'training': bool,
    'should_transpose_v': bool
  }

@trace
def _flash_attention_core(q_local_tile, k, v,
                          q_h_per_k_h,
                          o_buffer, l_buffer, m_buffer,
                          batch_id, head_id, gqa_head_idx, q_tile_idx,
                          local_k_large_tile_idx,
                          kernel_dtype, acc_type,
                          flash_config: FlashConfig,
                          olm_buffer_idx=None,
                          global_k_large_tile_idx=None,
                          use_causal_mask=False, initialize=False,
                          B_P_SIZE=128, B_F_SIZE=512, B_D_SIZE=128,
                          dropout_p=0.0, dropout_p_tensor=None, seed_tensor=None
                          ):
  """
  The flash attention core function to calcualte self attention between a tile of q and a block of K and V.
  The q_local_tile has (B_P_SIZE, B_F_SIZE), which is loaded into the SBUF already. The block size of K and V
  is defined in the seq_tile_size of the flash_config. The results are stored in the following there buffers
  o_buffer: (num_large_k_tile, B_P_SIZE, d)
  l_buffer: (num_large_k_tile, B_P_SIZE, 1)
  m_buffer: (num_large_k_tile, B_P_SIZE, 1)
  """
  LARGE_TILE_SZ = flash_config.seq_tile_size
  REDUCTION_TILE = min(2048, LARGE_TILE_SZ // 2)
  num_k_tile_per_large_tile = LARGE_TILE_SZ // B_F_SIZE
  seq_len = k.shape[-1]
  seq_q_num_tiles = seq_len // B_P_SIZE
  
  # Indices used by the distributed attention
  if global_k_large_tile_idx is None:
    global_k_large_tile_idx = local_k_large_tile_idx
  if olm_buffer_idx is None:
    olm_buffer_idx = local_k_large_tile_idx

  i_q_p = nl.arange(B_P_SIZE)[:, None]
  i_q_f = nl.arange(B_F_SIZE)[None, :]
  i_d_p = nl.arange(B_D_SIZE)[:, None]
  i_d_f = nl.arange(B_D_SIZE)[None, :]
  i_f_128 = nl.arange(B_P_SIZE)[None, :]
  i_f_k_tiles = nl.arange(num_k_tile_per_large_tile)[None, :]

  # mask are used to only apply computation to the lower half of the matrix, 
  # which reduce the arthimetic intensity by half
  forward_mask = q_tile_idx * B_P_SIZE >= global_k_large_tile_idx * LARGE_TILE_SZ if use_causal_mask else None
  # Negation mask is the negation of `forward_mask`, which is used for the 
  # instructions executed on the blocks in the upper triangular section 
  # of the matrix. 
  # These instructions should not be executed when causual mask is disabled.
  #
  # For example, the o_buffer still needs to be propagated from o[j-1] to o[j] in
  # the upper triangular of the matrix.
  negation_mask = q_tile_idx * B_P_SIZE < global_k_large_tile_idx * LARGE_TILE_SZ if use_causal_mask else None

  qk_res_buf = nl.ndarray((par_dim(B_P_SIZE), LARGE_TILE_SZ), buffer=nl.sbuf, dtype=acc_type)
  max_local = nl.ndarray((par_dim(B_P_SIZE), num_k_tile_per_large_tile), dtype=acc_type)
  for k_i in nl.affine_range(num_k_tile_per_large_tile):
    qk_psum = nl.zeros((par_dim(B_P_SIZE), B_F_SIZE),
                        dtype=np.float32, buffer=nl.psum)  # (128, 512)
    multiplication_required_selection = global_k_large_tile_idx * LARGE_TILE_SZ + k_i * B_F_SIZE <= q_tile_idx * B_P_SIZE if use_causal_mask else None
    qk_psum[i_q_p, i_q_f] += nl.matmul(q_local_tile, k[i_d_p, k_i * B_F_SIZE + i_q_f], transpose_x=True,
                                       mask=multiplication_required_selection) # (p(128), 512)
    
    if use_causal_mask:
      left_diagonal_selection = q_tile_idx * B_P_SIZE >= global_k_large_tile_idx * LARGE_TILE_SZ + (k_i + 1) * B_F_SIZE
      diagonal_and_right_selection = (q_tile_idx * B_P_SIZE < global_k_large_tile_idx * LARGE_TILE_SZ + (k_i + 1) * B_F_SIZE) & forward_mask

      q_pos = q_tile_idx * B_P_SIZE + i_q_p
      k_pos = global_k_large_tile_idx * LARGE_TILE_SZ + k_i * B_F_SIZE + i_q_f
      pred = q_pos >= k_pos
      # For tiles on and on the right of the diagonal, need to do affine_select.
      # Magic number -9984.0 to replace -inf similar to what Tensorizer uses
      qk_res_buf[i_q_p, k_i * B_F_SIZE + i_q_f] = nisa.affine_select(
        pred=pred,
        on_true_tile=qk_psum[i_q_p, i_q_f], on_false_value=-9984.0, dtype=kernel_dtype,
        mask=diagonal_and_right_selection)
      
      # For tiles on the left of the diagonal, direct copy, no select required.
      qk_res_buf[i_q_p, k_i * B_F_SIZE + i_q_f] = \
        nl.copy(qk_psum[i_q_p, i_q_f], dtype=kernel_dtype, mask=left_diagonal_selection)
    else:
      # Simply send psum result back to sbuf
      qk_res_buf[i_q_p, k_i * B_F_SIZE + i_q_f] = \
        nl.copy(qk_psum[i_q_p, i_q_f], dtype=kernel_dtype)

    # Calculate max of the current tile
    max_local[i_q_p, k_i] = nisa.reduce(np.max, qk_res_buf[i_q_p, k_i * B_F_SIZE + i_q_f], axis=(1,), 
                                        dtype=acc_type, negate=False, mask=forward_mask)

  max_ = nisa.reduce(np.max, max_local[i_q_p, i_f_k_tiles], axis=(1, ), 
                    dtype=acc_type, negate=False, mask=forward_mask)
  if not initialize:
    m_previous = nl.copy(m_buffer[olm_buffer_idx - 1, i_q_p, 0])
    m_buffer[olm_buffer_idx, i_q_p, 0] = nl.maximum(m_previous, max_, mask=forward_mask) # (128,1)
    if use_causal_mask:
      m_buffer[olm_buffer_idx, i_q_p, 0] = nl.copy(m_previous, mask=negation_mask)

    m_current = m_buffer[olm_buffer_idx, i_q_p, 0]
    # Compute scaling factor
    alpha = nisa.activation(np.exp, m_previous, bias=-1*m_current, scale=1.0, mask=forward_mask)
    o_previous = nl.copy(o_buffer[olm_buffer_idx-1, i_q_p, i_d_f], mask=forward_mask)
    o_previous_scaled = nl.multiply(o_previous, alpha, mask=forward_mask)
  else:
    m_buffer[0, i_q_p, 0] = nl.copy(max_)
    m_current = max_

  p_local = nl.ndarray((par_dim(B_P_SIZE), LARGE_TILE_SZ), dtype=kernel_dtype)
  i_r_f = nl.arange(REDUCTION_TILE)[None,: ]
  p_partial_sum = nl.ndarray((par_dim(B_P_SIZE), LARGE_TILE_SZ // REDUCTION_TILE), dtype=acc_type)
  for k_r_i in nl.affine_range(LARGE_TILE_SZ // REDUCTION_TILE):
    # compute exp(qk-max)
    p_local[i_q_p, k_r_i * REDUCTION_TILE + i_r_f] = \
      nisa.activation(np.exp,
                      qk_res_buf[i_q_p, k_r_i * REDUCTION_TILE + i_r_f],
                      bias=-1 * m_current,
                      scale=1.0,
                      dtype=kernel_dtype,
                      mask=forward_mask)
    
    # dropout
    if dropout_p > 0.0:
      for k_d_i in nl.sequential_range(REDUCTION_TILE // B_F_SIZE):
        offset = k_d_i + k_r_i * (REDUCTION_TILE // B_F_SIZE) \
                  + global_k_large_tile_idx * (LARGE_TILE_SZ // B_F_SIZE) \
                  + q_tile_idx * (seq_len // B_F_SIZE) \
                  + (head_id * q_h_per_k_h + gqa_head_idx) * (seq_len // B_F_SIZE) * seq_q_num_tiles \
                  + batch_id * nl.num_programs(1) * (seq_len // B_F_SIZE) * seq_q_num_tiles
        offset_seed = nl.add(seed_tensor[0, 0], offset, mask=forward_mask)
        nisa.random_seed(seed=offset_seed, mask=forward_mask)
        softmax_dropout = nl.dropout(p_local[i_q_p, k_r_i * REDUCTION_TILE + k_d_i * B_F_SIZE + i_q_f],
                                    rate=dropout_p_tensor[i_q_p, 0],
                                    mask=forward_mask)
        p_local[i_q_p, k_r_i * REDUCTION_TILE + k_d_i * B_F_SIZE + i_q_f] = \
          nl.multiply(softmax_dropout, 1 / (1 - dropout_p), mask=forward_mask)

    # Compute partial row-tile sum after exp(qk-max)
    p_partial_sum[i_q_p, k_r_i] = nl.sum(p_local[i_q_p, k_r_i * REDUCTION_TILE + i_r_f], axis=1, dtype=acc_type, mask=forward_mask)

  p_local_transposed = nl.ndarray((par_dim(B_P_SIZE), LARGE_TILE_SZ), dtype=kernel_dtype)
  for i_p_t in nl.affine_range(LARGE_TILE_SZ // 512):
    p_local_t_tmp = nl.ndarray((par_dim(B_P_SIZE), 512), buffer=nl.psum, dtype=np.float32)
    for i_p_t_local in nl.affine_range(512//128):
      p_local_t_tmp[i_q_p, i_p_t_local*128 + i_f_128] = nisa.sb_transpose(p_local[i_q_p, i_p_t*512+i_p_t_local * B_P_SIZE + i_f_128])
    i_f_512 = nl.arange(512)[None, :]
    p_local_transposed[i_q_p, i_p_t * 512 + i_f_512 ] = nl.copy(p_local_t_tmp[i_q_p, i_f_512], dtype=kernel_dtype)

  ps = nl.sum(p_partial_sum, axis=1, dtype=acc_type, mask=forward_mask)
  pv_psum = nl.zeros((par_dim(B_P_SIZE), B_D_SIZE), dtype=np.float32, buffer=nl.psum) # ideally we want to re-use qk_psum buffer here
  for k_i in nl.affine_range(LARGE_TILE_SZ // B_P_SIZE):
    pv_psum[i_q_p, i_d_f] += nl.matmul(p_local_transposed[i_q_p, k_i * B_P_SIZE + i_f_128],
                                       v[k_i, i_q_p, i_d_f],
                                       transpose_x=True,
                                       mask=forward_mask) # (128, 128) (p(Br), d)

  if initialize:
    o_buffer[olm_buffer_idx, i_q_p, i_d_f] = nl.copy(pv_psum[i_q_p, i_d_f])
    l_buffer[olm_buffer_idx, i_q_p, 0] = nl.add(nl.log(ps), max_)
  else:
    if use_causal_mask:
      o_buffer[olm_buffer_idx, i_q_p, i_d_f] = nl.copy(o_buffer[olm_buffer_idx-1, i_q_p, i_d_f], mask=negation_mask)
    o_buffer[olm_buffer_idx, i_q_p, i_d_f] = nl.add(o_previous_scaled, pv_psum, mask=forward_mask)
    
    l_prev = l_buffer[olm_buffer_idx-1, i_q_p, 0]
    l_exp = nl.add(nl.exp(nl.subtract(l_prev, m_current, mask=forward_mask), mask=forward_mask), ps, mask=forward_mask)
    l_buffer[olm_buffer_idx, i_q_p, 0] = nl.add(m_current, nl.log(l_exp, mask=forward_mask), mask=forward_mask)
    if use_causal_mask:
      l_buffer[olm_buffer_idx, i_q_p, 0] = nl.copy(l_buffer[olm_buffer_idx-1, i_q_p, 0], mask=negation_mask)


def flash_fwd(q, k, v, seed, o, lse=None,
              softmax_scale=None,
              use_causal_mask=True,
              mixed_precision=True,
              dropout_p=0.0, config: FlashConfig=FlashConfig()):
  """
  Inputs:
    q: query tensor of shape (b, h, d, seqlen)
    k: key tensor of shape (b, h, d, seqlen) or (b, kv_heads, d, seqlen)
    v: if config.should_transpose_v=False, value tensor of shape (b, h, seqlen, d) or (b, kv_heads, seqlen, d) 
       if config.should_transpose_v=True, value tensor of shape (b, h, d, seqlen) or (b, h, d, seqlen)
    seed: seed tensor of shape (1,)
  Outputs:
    o: output buffer of shape (b, h, seqlen, d)
    lse: log-sum-exp for bwd pass stored in (b, h, nl.tile_size.pmax, seqlen // nl.tile_size.pmax) where nl.tile_size.pmax is 128
  Compile-time Constants:
    softmax_scale: scaling for softmax, is None, default is `1.0/(d**0.5)`
    mixed_precision: flag to set non-matmul ops in fp32 precision, defualt is set to `true`, if false, we use same precision as input types
    causal_mask: flag to set causal masking
    config: dataclass with Performance config parameters for flash attention with default values
      seq_tile_size: `default=2048`, size of the kv tile size for attention computation
      reduction
      training: bool to indicate training vs inference `default=True`
  Performance Notes:cd
    For better performance, the kernel is tiled to be of size `LARGE_TILE_SZ`, and Flash attention math techniques are applied in unit
    of `LARGE_TILE_SZ`. Seqlen that is not divisible by `LARGE_TILE_SZ` is not supported at the moment.
  GQA support Notes: the spmd kernel for launching kernel should be on kv_heads instead of nheads
    ```
      e.g. 
      MHA: q: [b, h, d, s], k: [b, h, d,s] , v: [b,h, s, d]
        usage: flash_fwd[b,h](q,k,v,...)
      GQA: q: [b, h, d, s], k: [b, kv_h, d,s] , v: [b,kv_h, s, d]
        usage: flash_fwd[b,kv_h](q,k,v,...)
    ```
  """
  B_F_SIZE=512
  B_P_SIZE=128
  b , h, d, n  = q.shape
  assert isinstance(b, int)
  assert isinstance(h, int)
  assert isinstance(d, int)
  assert isinstance(n, int)
  use_causal_mask = True
  assert use_causal_mask==True
  mixed_precision=True
  B_D_SIZE = d
  k_h = k.shape[1]
  v_shape = v.shape
  if config.should_transpose_v:
    assert tuple(v_shape) == (b, k_h, d, n), f"V shape does not match layout requirements, expect: {(b, k_h, d, n)} but got {v_shape}"
    assert tuple(k.shape) == (b, k_h, d, n), f" k and v shape does not match the layout defined in the function, but got {k.shape}"
  else:
    assert tuple(v_shape) == (b, k_h, n, d), f"V shape does not match layout requirements, expect: {(b, k_h, n, d)} but got {v_shape}"
    assert k.shape == v_shape[:2] + v_shape[-1:] + v_shape[-2:-1], f" k and v shape does not match the layout defined in the function"
  assert d <= 128, f" we do not support head_dim > 128, got head dim {d}"
  kernel_dtype = nl.bfloat16 if mixed_precision else q.dtype
  acc_type =  np.dtype(np.float32) if mixed_precision else kernel_dtype
  
  i_q_p = nl.arange(B_P_SIZE)[:,None]
  i_0_f = nl.arange(1)[None, :]
  n_tile_q = n//B_P_SIZE # since q will be loaded on PE
  
  batch_id = nl.program_id(axis=0)
  head_id = nl.program_id(axis=1)
  #softmax_scale = softmax_scale or (1.0 / (d ** 0.5))
  softmax_scale = (1.0 / (d ** 0.5))

  LARGE_TILE_SZ = config.seq_tile_size
  assert config.seq_tile_size >= 512, f" seq tile_size {config.seq_tile_size} cannot be less than 512"
  assert n % LARGE_TILE_SZ == 0, f"seqlen is not divisible by {LARGE_TILE_SZ}"
  num_large_k_tile = n // LARGE_TILE_SZ

  REDUCTION_TILE = config.seq_tile_size // 2
  # inference flag, check if lse is none
  inference = not(config.training)
  if inference:
    assert lse is None, "lse should be none for inference"
    assert seed is None, f"seed should be None for inference, but got {seed}"
    assert dropout_p==0.0, f"dropout should be 0.0 for inference but got {dropout_p}"
  else:
    assert lse is not None, "lse should not be none for training"
  q_h_per_k_h = h // k_h

  if dropout_p > 0.0 and not inference:
    seed_local = nl.ndarray((par_dim(1), 1), buffer=nl.sbuf, dtype=nl.int32)
    seed_local[0, 0] = nl.load(seed[0])
    # TODO: Remove this once the dropout supports scale prob
    dropout_p_tensor = nl.full((B_P_SIZE, 1), fill_value=dropout_p, dtype=np.float32)
  else:
    dropout_p_tensor = None
    seed_local = None

  for i_q_h in nl.affine_range(q_h_per_k_h):

    # =============== Global Flash Attention accumulators ====================== #
    o_buffer = nl.full((n_tile_q, num_large_k_tile, par_dim(B_P_SIZE), d), 0.0, dtype=acc_type, buffer=nl.sbuf) # zeros does not work
    l_buffer = nl.full((n_tile_q, num_large_k_tile, par_dim(B_P_SIZE), 1), 0.0, dtype=acc_type, buffer=nl.sbuf)
    m_buffer = nl.full((n_tile_q, num_large_k_tile, par_dim(B_P_SIZE), 1), 0.0, dtype=acc_type) # workaround for nl.full issue
    # =============== Global Flash Attention accumulators END ================== #

    j = 0
    cur_k_tile = nl.ndarray((par_dim(B_D_SIZE), LARGE_TILE_SZ), dtype=kernel_dtype)
    cur_v_tile = nl.ndarray((LARGE_TILE_SZ//B_P_SIZE, par_dim(B_P_SIZE), B_D_SIZE), dtype=kernel_dtype)
    load_tile_size = B_P_SIZE
    for k_i in nl.affine_range(LARGE_TILE_SZ // load_tile_size):
      load_p = nl.arange(B_D_SIZE)[:, None]
      load_f = nl.arange(load_tile_size)[None, :]
      cur_k_tile[load_p, load_tile_size*k_i+load_f] = nl.load(
        k[batch_id, head_id, load_p, load_tile_size*k_i+load_f]
      )
    if config.should_transpose_v:
      for v_i in nl.affine_range(LARGE_TILE_SZ // load_tile_size):
        load_p = nl.arange(B_D_SIZE)[:, None]
        load_f = nl.arange(B_P_SIZE)[None, :]

        loaded = nl.load(v[batch_id, head_id, load_p, B_P_SIZE*v_i+load_f], dtype=kernel_dtype)
        store_p = nl.arange(B_P_SIZE)[:, None]
        store_f = nl.arange(B_D_SIZE)[None, :]
        cur_v_tile[v_i, store_p, store_f] = nisa.sb_transpose(loaded)
    else:
      for v_i in nl.affine_range(LARGE_TILE_SZ // load_tile_size):
        load_p = nl.arange(B_P_SIZE)[:, None]
        load_f = nl.arange(B_D_SIZE)[None, :]

        cur_v_tile[v_i, load_p, load_f] = nl.load(v[batch_id, head_id, B_P_SIZE*v_i+load_p, load_f], dtype=kernel_dtype)

    for i in nl.affine_range(n_tile_q):
      i_f_128 = nl.arange(B_P_SIZE)[None, :]
      i_f_d = nl.arange(B_D_SIZE)[None, :]
      i_p_d = nl.arange(B_D_SIZE)[:,None]
      q_tile = nl.ndarray((B_D_SIZE, B_P_SIZE),dtype=kernel_dtype)
      q_tile[i_p_d, i_f_128] = nl.load(q[batch_id, head_id * q_h_per_k_h + i_q_h, i_p_d, i*B_P_SIZE+i_f_128], dtype=kernel_dtype) \
                                * softmax_scale # load (d, 128) tile in SBUF
      _flash_attention_core(q_local_tile=q_tile, k=cur_k_tile, v=cur_v_tile,
                            q_h_per_k_h=q_h_per_k_h,
                            o_buffer=o_buffer[i], l_buffer=l_buffer[i], m_buffer=m_buffer[i],
                            batch_id=batch_id, head_id=head_id,
                            gqa_head_idx=i_q_h, q_tile_idx=i, local_k_large_tile_idx=0,
                            kernel_dtype=kernel_dtype, acc_type=acc_type,
                            flash_config=config, use_causal_mask=use_causal_mask,
                            initialize=True,
                            B_P_SIZE=B_P_SIZE, B_F_SIZE=B_F_SIZE, B_D_SIZE=B_D_SIZE,
                            dropout_p=dropout_p, dropout_p_tensor=dropout_p_tensor, seed_tensor=seed_local)

    for j in nl.sequential_range(1, num_large_k_tile):
      cur_k_tile = nl.ndarray((par_dim(B_D_SIZE), LARGE_TILE_SZ), dtype=kernel_dtype)
      cur_v_tile = nl.ndarray((LARGE_TILE_SZ//B_P_SIZE, par_dim(B_P_SIZE), B_D_SIZE), dtype=kernel_dtype)
      load_tile_size = B_P_SIZE
      for k_i in nl.affine_range(LARGE_TILE_SZ // load_tile_size):
        load_p = nl.arange(B_D_SIZE)[:, None]
        load_f = nl.arange(load_tile_size)[None, :]
        cur_k_tile[load_p, load_tile_size*k_i+load_f] = nl.load(
          k[batch_id, head_id, load_p, j*LARGE_TILE_SZ+load_tile_size*k_i+load_f]
        )
      if config.should_transpose_v:
        for v_i in nl.affine_range(LARGE_TILE_SZ // load_tile_size):
          load_p = nl.arange(B_D_SIZE)[:, None]
          load_f = nl.arange(B_P_SIZE)[None, :]

          loaded = nl.load(v[batch_id, head_id, load_p, j*LARGE_TILE_SZ+B_P_SIZE*v_i+load_f], dtype=kernel_dtype)
          store_p = nl.arange(B_P_SIZE)[:, None]
          store_f = nl.arange(B_D_SIZE)[None, :]
          cur_v_tile[v_i, store_p, store_f] = nisa.sb_transpose(loaded)
      else:
        for v_i in nl.affine_range(LARGE_TILE_SZ // load_tile_size):
          load_p = nl.arange(B_P_SIZE)[:, None]
          load_f = nl.arange(B_D_SIZE)[None, :]

          cur_v_tile[v_i, load_p, load_f] = nl.load(v[batch_id, head_id, j*LARGE_TILE_SZ+B_P_SIZE*v_i+load_p, load_f], dtype=kernel_dtype)

      for i in nl.affine_range(n_tile_q):
        i_f_128 = nl.arange(B_P_SIZE)[None, :]
        i_f_d = nl.arange(B_D_SIZE)[None, :]
        i_p_d = nl.arange(B_D_SIZE)[:,None]
        q_tile = nl.ndarray((B_D_SIZE, B_P_SIZE),dtype=kernel_dtype)
        q_tile[i_p_d, i_f_128] = nl.load(q[batch_id, head_id * q_h_per_k_h + i_q_h, i_p_d, i*B_P_SIZE+i_f_128], dtype=kernel_dtype) \
                                  * softmax_scale # load (d, 128) tile in SBUF
        _flash_attention_core(q_local_tile=q_tile, k=cur_k_tile, v=cur_v_tile,
                              q_h_per_k_h=q_h_per_k_h,
                              o_buffer=o_buffer[i], l_buffer=l_buffer[i], m_buffer=m_buffer[i],
                              batch_id=batch_id, head_id=head_id,
                              gqa_head_idx=i_q_h, q_tile_idx=i, local_k_large_tile_idx=j,
                              kernel_dtype=kernel_dtype, acc_type=acc_type,
                              flash_config=config, use_causal_mask=use_causal_mask,
                              initialize=False,
                              B_P_SIZE=B_P_SIZE, B_F_SIZE=B_F_SIZE, B_D_SIZE=B_D_SIZE,
                              dropout_p=dropout_p, dropout_p_tensor=dropout_p_tensor, seed_tensor=seed_local)

    # -------- write output to buffer on HBM ------------ #
    for i in nl.affine_range(n_tile_q):
      out = nl.ndarray((par_dim(B_P_SIZE), B_D_SIZE), dtype=kernel_dtype)
      out[i_q_p, i_f_d] = nl.multiply(o_buffer[i, num_large_k_tile - 1, i_q_p, i_f_d], 
                                      nl.exp(m_buffer[i, num_large_k_tile - 1, i_q_p, i_0_f] - l_buffer[i, num_large_k_tile - 1, i_q_p, i_0_f]),
                                      dtype=kernel_dtype)

      nl.store(o[batch_id, head_id * q_h_per_k_h + i_q_h, i * B_P_SIZE + i_q_p, i_f_d], out[i_q_p, i_f_d])
      if not inference:
        lse_local = nl.zeros((par_dim(B_P_SIZE), 1), dtype=acc_type)
        lse_local[i_q_p, i_0_f] = nl.copy(l_buffer[i, num_large_k_tile - 1, i_q_p, i_0_f], dtype=acc_type)
        nl.store(lse[batch_id, head_id * q_h_per_k_h + i_q_h, i_q_p, i + i_0_f], lse_local[i_q_p, i_0_f])


def attention_isa_kernel(q, k, v, scale, out, kernel_name: str):
  nisa.attention_kernel(kernel_name, q, k, v, scale, out)

def src_tgt_pairs_to_replica_groups(src_tgt_pairs):
  replica_group = set()
  for pair in src_tgt_pairs:
    replica_group.add(pair[0])
  return [sorted(replica_group)]

def src_tgt_pairs_to_ring_order(src_tgt_pairs, cur_rank):
  pair_map = {pair[1]: pair[0] for pair in src_tgt_pairs}
  num_workers = len(pair_map)
  ring_order = [cur_rank]
  next_rank = pair_map[cur_rank]
  for _ in range(num_workers-1):
    ring_order.append(next_rank)
    next_rank = pair_map[next_rank]
  if next_rank != cur_rank:
    raise ValueError(f"Invalid srt_tgt_pairs, got {src_tgt_pairs}")
  return ring_order

def ring_attention_bwd(
  q_ref, k_ref, v_ref, o_ref,
  dy_ref,
  lse_ref,
  seed_ref,
  out_dq_ref, out_dk_ref, out_dv_ref,
  rank_id=0,
  src_tgt_pairs=[],
  use_causal_mask=False,
  mixed_precision=False,
  dropout_p=0.0,
  softmax_scale=None,
):
  """
  Ring attention backward kernel. Compute the backward gradients with distributed inputs.

  IO tensor layouts:
   - q_ref: shape (bs, nheads, head_size, seq)
   - k_ref: shape (bs, nheads, head_size, seq)
   - v_ref: shape (bs, nheads, head_size, seq)
   - o_ref: shape (bs, nheads, head_size, seq)
   - dy_ref: shape (bs, nheads, head_size, seq)
   - lse_ref: shape (bs, nheads, nl.tile_size.pmax, seq // nl.tile_size.pmax)
   - out_dq_ref: shape (bs, nheads, head_size, seq)
   - out_dk_ref: shape (bs, nheads, head_size, seq)
   - out_dv_ref: shape (bs, nheads, head_size, seq)

  """

  # Use q_ref dtype as the intermediate tensor dtype
  # Assume all IO tensors have the same dtype
  kernel_dtype = q_ref.dtype
  mixed_dtype = np.dtype(np.float32) if mixed_precision else kernel_dtype

  replica_groups = src_tgt_pairs_to_replica_groups(src_tgt_pairs)
  ring_order = src_tgt_pairs_to_ring_order(src_tgt_pairs, rank_id)
  num_workers = len(src_tgt_pairs) if len(src_tgt_pairs) > 1 else 1 
  		
  assert q_ref.dtype == k_ref.dtype == v_ref.dtype == o_ref.dtype == dy_ref.dtype \
         == out_dq_ref.dtype == out_dk_ref.dtype == out_dv_ref.dtype
  assert lse_ref.dtype == mixed_dtype

  # Shape checking
  bs, nheads, d_head, seqlen = q_ref.shape
  assert tuple(k_ref.shape) == (bs, nheads, d_head, seqlen), \
    f"Input K shape mismatch, got {k_ref.shape}"
  assert tuple(v_ref.shape) == (bs, nheads, d_head, seqlen), \
    f"Input V shape mismatch, got {v_ref.shape}"
  assert tuple(o_ref.shape) == (bs, nheads, d_head, seqlen), \
    f"Input dy shape mismatch, got {o_ref.shape}"
  assert tuple(dy_ref.shape) == (bs, nheads, d_head, seqlen), \
    f"Input dy shape mismatch, got {dy_ref.shape}"
  assert tuple(lse_ref.shape) == (bs, nheads, nl.tile_size.pmax, seqlen // nl.tile_size.pmax), \
    f"Input lse shape mismatch, got {lse_ref.shape}"
  if seed_ref is not None:
    assert tuple(seed_ref.shape) == (1,), \
      f"Input seed shape mismatch, got {seed_ref.shape}"

  assert tuple(out_dq_ref.shape) == (bs, nheads, d_head, seqlen), \
    f"Output dQ shape mismatch, got {out_dq_ref.shape}"
  assert tuple(out_dk_ref.shape) == (bs, nheads, d_head, seqlen), \
    f"Output dK shape mismatch, got {out_dk_ref.shape}"
  assert tuple(out_dv_ref.shape) == (bs, nheads, d_head, seqlen), \
    f"Output dV shape mismatch, got {out_dv_ref.shape}"

  # FIXME: Add masking for different seqlen values.
  assert seqlen % 128 == 0, \
    f"Input sequence length must be divisible by 128, got {seqlen}"

  # Softmax scaling factor, multiplied onto Q
  softmax_scale = softmax_scale or 1.0 / float(d_head ** 0.5)

  # Different batch samples/attention heads have independent attention
  batch_id = nl.program_id(axis=0)
  head_id = nl.program_id(axis=1)

  q_seq_n_tiles, q_seq_tile_size = div_ceil(seqlen, 128), 128
  d_head_n_tiles, d_head_tile_size = div_ceil(d_head, 128), min(d_head, 128)

  if seqlen >= 512:
    k_seq_n_tiles, k_seq_tile_size = seqlen // 512, 512
  else:
    k_seq_n_tiles, k_seq_tile_size = seqlen // 128, 128

  k_seq_n_tiles_backward, k_seq_tile_size_backward = seqlen // 128, 128
  k_seq_fwd_bwd_tile_multipler = k_seq_tile_size // k_seq_tile_size_backward

  ##############################################################
  # Step 2.4 Prefetch exp bias for softmax
  ##############################################################
  softmax_exp_bias = nl.zeros((q_seq_n_tiles, par_dim(q_seq_tile_size), 1), dtype=mixed_dtype)
  for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):
    ip_qk = nl.arange(q_seq_tile_size)[:, None]
    lse_local = nl.load(
      lse_ref[batch_id, head_id, ip_qk, i_q_seq_tile],
      dtype=mixed_dtype)
    softmax_exp_bias[i_q_seq_tile, ip_qk, 0] = lse_local * -1.0

  ##############################################################
  # Step 1 Compute rowsum(dO ◦ O)
  ##############################################################
  dy_o_sum = nl.ndarray((q_seq_n_tiles, par_dim(q_seq_tile_size), 1), dtype=mixed_dtype)
  for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):
    ip_reduce = nl.arange(q_seq_tile_size)[:, None]
    dy_o_partial = nl.zeros((par_dim(q_seq_tile_size), d_head_n_tiles), dtype=mixed_dtype)
    for i_d_head_tile in nl.affine_range(d_head_n_tiles):
      ip_load = nl.arange(d_head_tile_size)[:, None]
      if_q = nl.arange(q_seq_tile_size)[None, :]
      dy_local = nl.load_transpose2d(
        dy_ref[batch_id, head_id, i_d_head_tile * d_head_tile_size + ip_load, i_q_seq_tile * q_seq_tile_size + if_q],
        dtype=mixed_dtype)
      o_local = nl.load_transpose2d(
        o_ref[batch_id, head_id, i_d_head_tile * d_head_tile_size + ip_load, i_q_seq_tile * q_seq_tile_size + if_q],
        dtype=mixed_dtype
      )

      dy_o_partial[ip_reduce, i_d_head_tile] = nisa.reduce(
        np.add, data=dy_local*o_local, axis=(1,), dtype=mixed_dtype
      )

    dy_o_sum[i_q_seq_tile, ip_reduce, 0] = nisa.reduce(
      np.add, data=dy_o_partial[ip_reduce, nl.arange(d_head_n_tiles)[None, :]],
      axis=(1,), dtype=mixed_dtype
    )

  # Indices for prefetch
  ip_qk = nl.arange(d_head_tile_size)[:, None]
  if_q = nl.arange(q_seq_tile_size)[None, :]
  if_k = nl.arange(k_seq_tile_size)[None, :]

  if dropout_p > 0.0:
    seed_local = nl.ndarray((par_dim(1), 1), buffer=nl.sbuf, dtype=nl.int32)
    seed_local[0, 0] = nl.load(seed_ref[0])
    # TODO: Remove this once the dropout supports scale prob
    dropout_p_local = nl.full((q_seq_tile_size, 1), fill_value=dropout_p, dtype=np.float32)
  else:
    seed_local = None
    dropout_p_local = None

  dq_local_reduced = nl.zeros((q_seq_n_tiles, d_head_n_tiles, par_dim(d_head_tile_size), q_seq_tile_size),
                              dtype=mixed_dtype)

  # Local buffer to hold dK and dV
  dk_acc_buf = nl.ndarray((d_head, seqlen), dtype=nl.float32, buffer=nl.private_hbm, name="dk_acc_buf")
  dv_acc_buf = nl.ndarray((d_head, seqlen), dtype=nl.float32, buffer=nl.private_hbm, name="dv_acc_buf")

  # Double buffer to hold the Q, dy and dy_o_sum
  send_q_buf = nl.ndarray((d_head, seqlen), dtype=q_ref.dtype, buffer=nl.private_hbm, name="send_q_buf")
  recv_q_buf = nl.ndarray((d_head, seqlen), dtype=q_ref.dtype, buffer=nl.private_hbm, name="recv_q_buf")
  send_dy_buf = nl.ndarray((d_head, seqlen), dtype=dy_ref.dtype, buffer=nl.private_hbm, name="send_dy_buf")
  recv_dy_buf = nl.ndarray((d_head, seqlen), dtype=dy_ref.dtype, buffer=nl.private_hbm, name="recv_dy_buf")
  # send_dy_o_sum_buf = nl.ndarray((q_seq_n_tiles, par_dim(q_seq_tile_size), 1), dtype=dy_o_sum.dtype, buffer=nl.private_hbm, name="send_dy_o_sum_buf")
  # recv_dy_o_sum_buf = nl.ndarray((q_seq_n_tiles, par_dim(q_seq_tile_size), 1), dtype=dy_o_sum.dtype, buffer=nl.private_hbm, name="recv_dy_o_sum_buf")
  send_dy_o_sum_buf = nl.ndarray((seqlen, 1), dtype=dy_o_sum.dtype, buffer=nl.private_hbm, name="send_dy_o_sum_buf")
  recv_dy_o_sum_buf = nl.ndarray((seqlen, 1), dtype=dy_o_sum.dtype, buffer=nl.private_hbm, name="recv_dy_o_sum_buf")  
  send_dq_buf = nl.ndarray((d_head, seqlen), dtype=nl.float32, buffer=nl.private_hbm, name="send_dq_buf")
  recv_dq_buf = nl.ndarray((d_head, seqlen), dtype=nl.float32, buffer=nl.private_hbm, name="recv_dq_buf")

  ip_send_q_buf, if_send_q_buf = nl.mgrid[0:d_head_tile_size, 0:seqlen]
  ip_dy_o_sum_buf, if_dy_o_sum_buf = nl.mgrid[0:q_seq_tile_size, 0:1]

  # Initialize the buffer
  for i_d_head_tile in nl.affine_range(d_head_n_tiles):
    nisa._tiled_offloaded_memcpy(dst=send_q_buf[i_d_head_tile * d_head_tile_size + ip_send_q_buf, if_send_q_buf],
                                 src=q_ref[batch_id, head_id, i_d_head_tile * d_head_tile_size + ip_send_q_buf, if_send_q_buf])
    nisa._tiled_offloaded_memcpy(dst=send_dy_buf[i_d_head_tile * d_head_tile_size + ip_send_q_buf, if_send_q_buf],
                                 src=dy_ref[batch_id, head_id, i_d_head_tile * d_head_tile_size + ip_send_q_buf, if_send_q_buf])

  for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):
    nl.store(dst=send_dy_o_sum_buf[i_q_seq_tile * q_seq_tile_size + ip_dy_o_sum_buf, if_dy_o_sum_buf],
             value=dy_o_sum[i_q_seq_tile, ip_dy_o_sum_buf, if_dy_o_sum_buf])

  # Send the local buffers to the neighbors
  def _collective_permute_buffers():
    # for i_d_head_tile in nl.affine_range(d_head_n_tiles):
    nccl.collective_permute(dst=recv_q_buf[:, :], src=send_q_buf[:, :],
                                  replica_groups=replica_groups)
    nccl.collective_permute(dst=recv_dy_buf[:, :], src=send_dy_buf[:, :],
                                  replica_groups=replica_groups)
    nccl.collective_permute(dst=recv_dy_o_sum_buf[:, :], src=send_dy_o_sum_buf[:, :],
                                   replica_groups=replica_groups)
  _collective_permute_buffers()

  # affine_range give the compiler permission to vectorize instructions
  # inside the loop which improves the performance. However, when using the 
  # the dropout we should use sequential_range to avoid setting
  # seed vectorization. TODO: the compiler should avoid vectorizing seed setting
  _range = nl.sequential_range if dropout_p > 0.0 else nl.affine_range
  
  # Calculate the gradients based on the local Q and dy
  for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
    # Prefetch V, K
    v_local = nl.zeros((d_head_n_tiles, par_dim(d_head_tile_size), k_seq_tile_size), dtype=kernel_dtype)
    k_local = nl.zeros((d_head_n_tiles, par_dim(d_head_tile_size), k_seq_tile_size), dtype=kernel_dtype)
    transposed_k_local = nl.zeros((k_seq_fwd_bwd_tile_multipler, d_head_n_tiles, par_dim(k_seq_tile_size_backward), d_head_tile_size), dtype=kernel_dtype)
    for i_d_head_tile in nl.affine_range(d_head_n_tiles):
      k_local[i_d_head_tile, ip_qk, if_k] = nl.load(
        k_ref[batch_id, head_id, i_d_head_tile * d_head_tile_size + ip_qk, i_k_seq_tile * k_seq_tile_size + if_k],
        dtype=kernel_dtype)
      v_local[i_d_head_tile, ip_qk, if_k] = nl.load(
        v_ref[batch_id, head_id, i_d_head_tile * d_head_tile_size + ip_qk, i_k_seq_tile * k_seq_tile_size + if_k],
        dtype=kernel_dtype)
      ##############################################################
      # Prefetch k transpose for the backward too
      ##############################################################
      if_k_backward = nl.arange(k_seq_tile_size_backward)[None, :]
      ip_k_backward = nl.arange(k_seq_tile_size_backward)[:, None]
      if_d_head = nl.arange(d_head_tile_size)[None, :]
      for i_k_seq_tile_backward in nl.affine_range(k_seq_fwd_bwd_tile_multipler):
        transposed_k_local[i_k_seq_tile_backward, i_d_head_tile, ip_k_backward, if_d_head] = \
          nisa.sb_transpose(k_local[i_d_head_tile, ip_qk,
                                    i_k_seq_tile_backward * k_seq_tile_size_backward + if_k_backward])

    dv_psum = nl.zeros((d_head_n_tiles, par_dim(d_head_tile_size), k_seq_tile_size),
                        dtype=np.float32, buffer=nl.psum)
    dk_psum = nl.zeros((d_head_n_tiles, par_dim(d_head_tile_size), k_seq_tile_size),
                        dtype=np.float32, buffer=nl.psum)
    for i_q_seq_tile in _range(q_seq_n_tiles):
      # Prefetch dy, Q
      dy_local = nl.zeros((d_head_n_tiles, par_dim(d_head_tile_size), q_seq_tile_size), dtype=kernel_dtype)
      q_local = nl.zeros((d_head_n_tiles, par_dim(d_head_tile_size), q_seq_tile_size), dtype=kernel_dtype)
      for i_d_head_tile in nl.affine_range(d_head_n_tiles):
        ip_qk = nl.arange(d_head_tile_size)[:, None]
        if_q = nl.arange(q_seq_tile_size)[None, :]

        dy_local[i_d_head_tile, ip_qk, if_q] = nl.load(
          dy_ref[batch_id, head_id, i_d_head_tile * d_head_tile_size + ip_qk, i_q_seq_tile * q_seq_tile_size + if_q],
          dtype=kernel_dtype)

        q_local[i_d_head_tile, ip_qk, if_q] = nl.load(
          q_ref[batch_id, head_id, i_d_head_tile * d_head_tile_size + ip_qk, i_q_seq_tile * q_seq_tile_size + if_q],
          dtype=kernel_dtype) * softmax_scale

      _flash_attn_bwd_core(
        q_local=q_local, k_local=k_local, transposed_k_local=transposed_k_local,
        v_local=v_local, dy_local=dy_local,
        dk_psum=dk_psum, dv_psum=dv_psum, dq_local_reduced=dq_local_reduced,
        softmax_exp_bias=softmax_exp_bias, dy_o_sum=dy_o_sum,
        local_i_q_seq_tile=i_q_seq_tile, local_i_k_seq_tile=i_k_seq_tile,
        global_i_q_seq_tile=rank_id * q_seq_n_tiles + i_q_seq_tile,
        global_i_k_seq_tile=rank_id * k_seq_n_tiles + i_k_seq_tile,
        seqlen=seqlen, d_head=d_head, 
        use_causal_mask=use_causal_mask,
        kernel_dtype=kernel_dtype, mixed_dtype=mixed_dtype,
        softmax_scale=softmax_scale,
        seed_local=seed_local, dropout_p=dropout_p, dropout_p_local=dropout_p_local,       
      )

    # Write dK, dV
    for i_d_head_tile in nl.affine_range(d_head_n_tiles):
      ip_dkv = nl.arange(d_head_tile_size)[:, None]
      if_dkv = nl.arange(k_seq_tile_size)[None, :]

      nl.store(
        dv_acc_buf[i_d_head_tile * d_head_tile_size + ip_dkv,
                   i_k_seq_tile * k_seq_tile_size + if_dkv],
        value=dv_psum[i_d_head_tile, ip_dkv, if_dkv],
      )

      nl.store(
        dk_acc_buf[i_d_head_tile * d_head_tile_size + ip_dkv,
                   i_k_seq_tile * k_seq_tile_size + if_dkv],
        value=dk_psum[i_d_head_tile, ip_dkv, if_dkv],
      )

  # Write dq to local buffer and set to next neighbor
  for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):
    for i_d_head_tile in nl.affine_range(d_head_n_tiles):
      ip_dq = nl.arange(d_head_tile_size)[:, None]
      if_dq = nl.arange(q_seq_tile_size)[None, :]

      nl.store(
        send_dq_buf[i_d_head_tile * d_head_tile_size + ip_dq,
                    i_q_seq_tile * q_seq_tile_size + if_dq],
        value=dq_local_reduced[i_q_seq_tile, i_d_head_tile, ip_dq, if_dq],
      )

  nccl.collective_permute(dst=recv_dq_buf[:, :], src=send_dq_buf[:, :],
                          replica_groups=replica_groups)
  # Swap the buffer
  def _swap_buffer():
    nisa._tiled_offloaded_memcpy(dst=send_q_buf[:, :], src=recv_q_buf[:, :])
    nisa._tiled_offloaded_memcpy(dst=send_dy_buf[:, :], src=recv_dy_buf[:, :])

    nisa._tiled_offloaded_memcpy(dst=send_dy_o_sum_buf[:, :],  src=recv_dy_o_sum_buf[:, :])
    for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):
      dy_o_sum[i_q_seq_tile, ip_dy_o_sum_buf, if_dy_o_sum_buf] = \
        nl.load(send_dy_o_sum_buf[i_q_seq_tile * q_seq_tile_size + ip_dy_o_sum_buf, if_dy_o_sum_buf])

  _swap_buffer()

  # Keep receiving the q, dy from neighbors
  # TODO: Use sequential_range
  for ring_step in nl.static_range(1, num_workers):
    ring_rank_id = ring_order[ring_step]
    dk_buf = nl.ndarray((d_head, seqlen), dtype=nl.float32, buffer=nl.private_hbm, name="dk_buf")
    dv_buf = nl.ndarray((d_head, seqlen), dtype=nl.float32, buffer=nl.private_hbm, name="dv_buf")
    dq_local_reduced = nl.zeros((q_seq_n_tiles, d_head_n_tiles, par_dim(d_head_tile_size), q_seq_tile_size),
                                dtype=mixed_dtype)

    _collective_permute_buffers()

    # Calculate the gradients based on the local Q and dy
    for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
      # Prefetch V, K
      v_local = nl.zeros((d_head_n_tiles, par_dim(d_head_tile_size), k_seq_tile_size), dtype=kernel_dtype)
      k_local = nl.zeros((d_head_n_tiles, par_dim(d_head_tile_size), k_seq_tile_size), dtype=kernel_dtype)
      transposed_k_local = nl.zeros((k_seq_fwd_bwd_tile_multipler, d_head_n_tiles, par_dim(k_seq_tile_size_backward), d_head_tile_size), dtype=kernel_dtype)
      for i_d_head_tile in nl.affine_range(d_head_n_tiles):
        k_local[i_d_head_tile, ip_qk, if_k] = nl.load(
          k_ref[batch_id, head_id, i_d_head_tile * d_head_tile_size + ip_qk, i_k_seq_tile * k_seq_tile_size + if_k],
          dtype=kernel_dtype)
        v_local[i_d_head_tile, ip_qk, if_k] = nl.load(
          v_ref[batch_id, head_id, i_d_head_tile * d_head_tile_size + ip_qk, i_k_seq_tile * k_seq_tile_size + if_k],
          dtype=kernel_dtype)
        ##############################################################
        # Prefetch k transpose for the backward too
        ##############################################################
        if_k_backward = nl.arange(k_seq_tile_size_backward)[None, :]
        ip_k_backward = nl.arange(k_seq_tile_size_backward)[:, None]
        if_d_head = nl.arange(d_head_tile_size)[None, :]
        for i_k_seq_tile_backward in nl.affine_range(k_seq_fwd_bwd_tile_multipler):
          transposed_k_local[i_k_seq_tile_backward, i_d_head_tile, ip_k_backward, if_d_head] = \
            nisa.sb_transpose(k_local[i_d_head_tile, ip_qk,
                                      i_k_seq_tile_backward * k_seq_tile_size_backward + if_k_backward])

      dv_psum = nl.zeros((d_head_n_tiles, par_dim(d_head_tile_size), k_seq_tile_size),
                          dtype=np.float32, buffer=nl.psum)
      dk_psum = nl.zeros((d_head_n_tiles, par_dim(d_head_tile_size), k_seq_tile_size),
                          dtype=np.float32, buffer=nl.psum)
      for i_q_seq_tile in _range(q_seq_n_tiles):
        # Prefetch dy, Q
        dy_local = nl.zeros((d_head_n_tiles, par_dim(d_head_tile_size), q_seq_tile_size), dtype=kernel_dtype)
        q_local = nl.zeros((d_head_n_tiles, par_dim(d_head_tile_size), q_seq_tile_size), dtype=kernel_dtype)
        for i_d_head_tile in nl.affine_range(d_head_n_tiles):
          ip_qk = nl.arange(d_head_tile_size)[:, None]
          if_q = nl.arange(q_seq_tile_size)[None, :]

          dy_local[i_d_head_tile, ip_qk, if_q] = nl.load(
            send_dy_buf[i_d_head_tile * d_head_tile_size + ip_qk, i_q_seq_tile * q_seq_tile_size + if_q],
            dtype=kernel_dtype)

          q_local[i_d_head_tile, ip_qk, if_q] = nl.load(
            send_q_buf[i_d_head_tile * d_head_tile_size + ip_qk, i_q_seq_tile * q_seq_tile_size + if_q],
            dtype=kernel_dtype) * softmax_scale

        _flash_attn_bwd_core(
          q_local=q_local, k_local=k_local, transposed_k_local=transposed_k_local,
          v_local=v_local, dy_local=dy_local,
          dk_psum=dk_psum, dv_psum=dv_psum, dq_local_reduced=dq_local_reduced,
          softmax_exp_bias=softmax_exp_bias, dy_o_sum=dy_o_sum,
          local_i_q_seq_tile=i_q_seq_tile, local_i_k_seq_tile=i_k_seq_tile,
          global_i_q_seq_tile=ring_rank_id * q_seq_n_tiles + i_q_seq_tile,
          global_i_k_seq_tile=rank_id * k_seq_n_tiles + i_k_seq_tile,
          seqlen=seqlen, d_head=d_head, 
          use_causal_mask=use_causal_mask,
          kernel_dtype=kernel_dtype, mixed_dtype=mixed_dtype,
          softmax_scale=softmax_scale,
          seed_local=seed_local, dropout_p=dropout_p, dropout_p_local=dropout_p_local,       
        )

      # Write dK, dV
      for i_d_head_tile in nl.affine_range(d_head_n_tiles):
        ip_dkv = nl.arange(d_head_tile_size)[:, None]
        if_dkv = nl.arange(k_seq_tile_size)[None, :]

        nl.store(
          dv_buf[i_d_head_tile * d_head_tile_size + ip_dkv,
                 i_k_seq_tile * k_seq_tile_size + if_dkv],
          value=dv_psum[i_d_head_tile, ip_dkv, if_dkv],
        )

        nl.store(
          dk_buf[i_d_head_tile * d_head_tile_size + ip_dkv,
                 i_k_seq_tile * k_seq_tile_size + if_dkv],
          value=dk_psum[i_d_head_tile, ip_dkv, if_dkv],
        )

    
    # Write dq to local buffer and send to next neighbor
    dq_add_tmp_buf = nl.ndarray((d_head, seqlen), dtype=nl.float32, buffer=nl.private_hbm, name="dq_add_tmp_buf")
    for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):
      for i_d_head_tile in nl.affine_range(d_head_n_tiles):
        ip_dq = nl.arange(d_head_tile_size)[:, None]
        if_dq = nl.arange(q_seq_tile_size)[None, :]

        nl.store(
          dq_add_tmp_buf[i_d_head_tile * d_head_tile_size + ip_dq,
                         i_q_seq_tile * q_seq_tile_size + if_dq],
          value=dq_local_reduced[i_q_seq_tile, i_d_head_tile, ip_dq, if_dq],
        )

    nisa._tiled_offloaded_fma(
      dq_add_tmp_buf[:, :], recv_dq_buf[:, :],
      scales=[1.0, 1.0],
      dst=send_dq_buf[:, :]
    )
    nccl.collective_permute(dst=recv_dq_buf[:, :], src=send_dq_buf[:, :],
                            replica_groups=replica_groups)


    dk_fma_tmp_buf = nl.ndarray((d_head, seqlen), dtype=nl.float32, buffer=nl.private_hbm, name="dk_fma_tmp_buf")
    dv_fma_tmp_buf = nl.ndarray((d_head, seqlen), dtype=nl.float32, buffer=nl.private_hbm, name="dv_fma_tmp_buf")
    # Accumulate the dK dV results
    nisa._tiled_offloaded_fma(
      dk_buf[:, :], dk_acc_buf[:, :],
      scales=[1.0, 1.0],
      dst=dk_fma_tmp_buf[:, :],
    )
    nisa._tiled_offloaded_memcpy(
      dst=dk_acc_buf[:, :], src=dk_fma_tmp_buf[:, :]
    )
    nisa._tiled_offloaded_fma(
      dv_buf[:, :], dv_acc_buf[:, :],
      scales=[1.0, 1.0],
      dst=dv_fma_tmp_buf[:, :],
    )
    nisa._tiled_offloaded_memcpy(
      dst=dv_acc_buf[:, :], src=dv_fma_tmp_buf[:, :]
    )

    # Swap the buffer
    _swap_buffer() 

  # Write to final output dK, dV
  for i_d_head_tile in nl.affine_range(d_head_n_tiles):
    ip_dkv = nl.arange(d_head_tile_size)[:, None]
    if_seq = nl.arange(seqlen)[None, :]

    nisa._tiled_offloaded_memcpy(
      dst=out_dv_ref[batch_id, head_id,
                     i_d_head_tile * d_head_tile_size + ip_dkv, if_seq],
      src=dv_acc_buf[i_d_head_tile * d_head_tile_size + ip_dkv, if_seq],
      dtype=out_dv_ref.dtype
    )

    nisa._tiled_offloaded_memcpy(
      dst=out_dk_ref[batch_id, head_id,
                     i_d_head_tile * d_head_tile_size + ip_dkv, if_seq],
      src=dk_acc_buf[i_d_head_tile * d_head_tile_size + ip_dkv, if_seq],
      dtype=out_dk_ref.dtype
    )

  # Write dQ
  ip_dq_ref, if_dq_ref = nl.mgrid[0:d_head, 0:seqlen]
  nisa._tiled_offloaded_memcpy(
    dst=out_dq_ref[batch_id, head_id, ip_dq_ref, if_dq_ref],
    src=send_dq_buf[:, :],
    dtype=out_dq_ref.dtype
  )
  
def ring_attention_fwd(q, k, v, seed, o, lse=None,
                       rank_id=0,
                       src_tgt_pairs=[],
                       softmax_scale=None,
                       use_causal_mask=True,
                       mixed_precision=True,
                       dropout_p=0.0,
                       config: FlashConfig=FlashConfig()):
  """
  The NKI ring attention implementation on top of the flash attention.

  Inputs:
    q: query tensor of shape (b, h, d, seqlen_chunk)
    k: key tensor of shape (b, kv_heads, d, seqlen_chunk)
    v: value tensor of shape (b, kv_heads, d, seqlen_chunk)
    seed: seed tensor of shape (1,)
  
  Outputs:
    o: output buffer of shape (b, h, seqlen, d)
    lse: log-sum-exp for bwd pass stored in (b, h, nl.tile_size.pmax, seqlen // nl.tile_size.pmax) where nl.tile_size.pmax is 128
  
  Compile-time Constants:
    rank_id: The current worker rank, important when we use causal mask.
    src_tgt_paris: The list describing the ring to communication. 
    softmax_scale: scaling for softmax, is None, default is `1.0/(d**0.5)`
    mixed_precision: flag to set non-matmul ops in fp32 precision, defualt is set to `true`, if false, we use same precision as input types
    causal_mask: flag to set causal masking
    config: dataclass with Performance config parameters for flash attention with default values
      seq_tile_size: `default=2048`, size of the kv tile size for attention computation
      reduction
      training: bool to indicate training vs inference `default=True`

  Performance Notes:
    For better performance, the kernel is tiled to be of size `LARGE_TILE_SZ`, and Flash attention math techniques are applied in unit
    of `LARGE_TILE_SZ`. Seqlen that is not divisible by `LARGE_TILE_SZ` is not supported at the moment.
  GQA support Notes: the spmd kernel for launching kernel should be on kv_heads instead of nheads
    ```
      e.g. 
      MHA: q: [b, h, d, s], k: [b, h, d, s] , v: [b, h, s, d]
        usage: flash_fwd[b, h](q, k, v,...)
      GQA: q: [b, h, d, s], k: [b, kv_h, d, s] , v: [b, kv_h, s, d]
        usage: flash_fwd[b, kv_h](q, k, v,...)
    ```
  """
  B_F_SIZE=512
  B_P_SIZE=128
  b , h, d, n  = q.shape
  B_D_SIZE = d
  k_h = k.shape[1]
  v_shape = v.shape
  assert config.should_transpose_v, f" require to use set the should_transpose_v in the FlashConfig"
  assert tuple(v_shape) == (b, k_h, d, n), f"V shape does not match layout requirements, expect: {(b, k_h, d, n)} but got {v_shape}"
  assert tuple(k.shape) == (b, k_h, d, n), f" k and v shape does not match the layout defined in the function, but got {k.shape}"
  assert d <= 128, f" we do not support head_dim > 128, got head dim {d}"
  assert use_causal_mask, f" use without causal mask is not tested yet. "
  kernel_dtype = nl.bfloat16 if mixed_precision else q.dtype
  acc_type =  np.dtype(np.float32) if mixed_precision else kernel_dtype
  num_workers = len(src_tgt_pairs) if len(src_tgt_pairs) > 1 else 1 
  replica_groups = src_tgt_pairs_to_replica_groups(src_tgt_pairs)
  ring_order = src_tgt_pairs_to_ring_order(src_tgt_pairs, rank_id)

  n_tile_q = n // B_P_SIZE # since q will be loaded on PE
  q_h_per_k_h = h // k_h
  softmax_scale = softmax_scale or (1.0 / (d ** 0.5))

  LARGE_TILE_SZ = config.seq_tile_size
  assert config.seq_tile_size >= 512, f" seq tile_size {config.seq_tile_size} cannot be less than 512"
  assert n % LARGE_TILE_SZ == 0, f"seqlen is not divisible by {LARGE_TILE_SZ}"
  num_large_k_tile = n // LARGE_TILE_SZ
  REDUCTION_TILE = config.seq_tile_size // 2
  
  # inference flag, check if lse is none
  inference = not(config.training)
  if inference:
    assert lse is None, "lse should be none for inference"
    assert seed is None, f"seed should be None for inference, but got {seed}"
    assert dropout_p == 0.0, f"dropout should be 0.0 for inference but got {dropout_p}"
  else:
    assert lse is not None, "lse should not be none for training"
  
  if dropout_p > 0.0 and not inference:
    seed_local = nl.ndarray((par_dim(1), 1), buffer=nl.sbuf, dtype=nl.int32)
    seed_local[0, 0] = nl.load(seed[0])
    # TODO: Remove this once the dropout supports scale prob
    dropout_p_tensor = nl.full((B_P_SIZE, 1), fill_value=dropout_p, dtype=np.float32)
  else:
    dropout_p_tensor = None
    seed_local = None

  batch_id = nl.program_id(axis=0)
  head_id = nl.program_id(axis=1)

  # Virtual global flash attention accumulators
  o_buffer = nl.full((q_h_per_k_h, n_tile_q, num_large_k_tile * num_workers, par_dim(B_P_SIZE), d), 0.0,
                    dtype=acc_type, buffer=nl.sbuf, name="o_buffer") # zeros does not work
  l_buffer = nl.full((q_h_per_k_h, n_tile_q, num_large_k_tile * num_workers, par_dim(B_P_SIZE), 1), 0.0,
                    dtype=acc_type, buffer=nl.sbuf, name="l_buffer")
  m_buffer = nl.full((q_h_per_k_h, n_tile_q, num_large_k_tile * num_workers, par_dim(B_P_SIZE), 1), 0.0,
                    dtype=acc_type, buffer=nl.sbuf, name="m_buffer")

  # Double buffers to hold the sharded KV values
  send_k_buf = nl.ndarray((par_dim(d), n), dtype=k.dtype, buffer=nl.private_hbm, name="send_k_buf")
  recv_k_buf = nl.ndarray((par_dim(d), n), dtype=k.dtype, buffer=nl.private_hbm, name="recv_k_buf")
  send_v_buf = nl.ndarray((par_dim(d), n), dtype=v.dtype, buffer=nl.private_hbm, name="send_v_buf")
  recv_v_buf = nl.ndarray((par_dim(d), n), dtype=v.dtype, buffer=nl.private_hbm, name="recv_v_buf")

  # kv_idx, kv_buf_ix, kv_buf_iy = nl.mgrid[0:2, 0:d, 0:n]
  # kv_idx = nl.arange(2)[None, :, None]
  kv_buf_ix = nl.arange(d)[:, None]
  kv_buf_iy = nl.arange(n)[None, :]

  # Initialize the buffer
  nisa._tiled_offloaded_memcpy(dst=send_k_buf[kv_buf_ix, kv_buf_iy],
                               src=k[batch_id, head_id, kv_buf_ix, kv_buf_iy])
  nisa._tiled_offloaded_memcpy(dst=send_v_buf[kv_buf_ix, kv_buf_iy],
                               src=v[batch_id, head_id, kv_buf_ix, kv_buf_iy])
  nccl.collective_permute(src=send_k_buf[kv_buf_ix, kv_buf_iy],
                          dst=recv_k_buf[kv_buf_ix, kv_buf_iy],
                          replica_groups=replica_groups)
  nccl.collective_permute(src=send_v_buf[kv_buf_ix, kv_buf_iy],
                          dst=recv_v_buf[kv_buf_ix, kv_buf_iy],
                          replica_groups=replica_groups)

  i_f_128 = nl.arange(B_P_SIZE)[None, :]
  i_f_d = nl.arange(B_D_SIZE)[None, :]
  i_p_d = nl.arange(B_D_SIZE)[:,None]
  i_q_p = nl.arange(B_P_SIZE)[:,None]
  i_0_f = nl.arange(1)[None, :]

  # First calculate the local self-attention
  for i_q_h in nl.affine_range(q_h_per_k_h):
    cur_k_tile = nl.ndarray((par_dim(B_D_SIZE), LARGE_TILE_SZ), dtype=kernel_dtype)
    cur_v_tile = nl.ndarray((LARGE_TILE_SZ // B_P_SIZE, par_dim(B_P_SIZE), B_D_SIZE), dtype=kernel_dtype)
    load_tile_size = B_P_SIZE
    for k_i in nl.affine_range(LARGE_TILE_SZ // load_tile_size):
      load_p, load_f = nl.mgrid[0:B_D_SIZE, 0:load_tile_size]
      cur_k_tile[load_p, load_tile_size * k_i + load_f] = nl.load(
        k[batch_id, head_id, load_p, load_tile_size * k_i + load_f]
      )
    for v_i in nl.affine_range(LARGE_TILE_SZ // load_tile_size):
      load_p, load_f = nl.mgrid[0:B_D_SIZE, 0:B_P_SIZE]
      store_p, store_f = nl.mgrid[0:B_P_SIZE, 0:B_D_SIZE]
      loaded = nl.load(v[batch_id, head_id, load_p, B_P_SIZE * v_i + load_f], dtype=kernel_dtype)
      cur_v_tile[v_i, store_p, store_f] = nisa.sb_transpose(loaded)

    for i in nl.affine_range(n_tile_q):
      q_tile = nl.ndarray((B_D_SIZE, B_P_SIZE),dtype=kernel_dtype)
      q_tile[i_p_d, i_f_128] = nl.load(q[batch_id, head_id * q_h_per_k_h + i_q_h, i_p_d, i * B_P_SIZE + i_f_128], dtype=kernel_dtype) \
                                * softmax_scale # load (d, 128) tile in SBUF
      _flash_attention_core(q_local_tile=q_tile, k=cur_k_tile, v=cur_v_tile,
                            q_h_per_k_h=q_h_per_k_h,
                            o_buffer=o_buffer[i_q_h, i], l_buffer=l_buffer[i_q_h, i], m_buffer=m_buffer[i_q_h, i],
                            batch_id=batch_id, head_id=head_id,
                            gqa_head_idx=i_q_h, q_tile_idx=rank_id * n_tile_q + i,
                            global_k_large_tile_idx=rank_id * num_large_k_tile,
                            local_k_large_tile_idx=0,
                            olm_buffer_idx=0,
                            kernel_dtype=kernel_dtype, acc_type=acc_type,
                            flash_config=config, use_causal_mask=use_causal_mask,
                            initialize=True,
                            B_P_SIZE=B_P_SIZE, B_F_SIZE=B_F_SIZE, B_D_SIZE=B_D_SIZE,
                            dropout_p=dropout_p, dropout_p_tensor=dropout_p_tensor, seed_tensor=seed_local)

    for j in nl.sequential_range(1, num_large_k_tile):
      cur_k_tile = nl.ndarray((par_dim(B_D_SIZE), LARGE_TILE_SZ), dtype=kernel_dtype)
      cur_v_tile = nl.ndarray((LARGE_TILE_SZ//B_P_SIZE, par_dim(B_P_SIZE), B_D_SIZE), dtype=kernel_dtype)
      load_tile_size = B_P_SIZE
      for k_i in nl.affine_range(LARGE_TILE_SZ // load_tile_size):
        load_p, load_f = nl.mgrid[0:B_D_SIZE, 0:load_tile_size]
        cur_k_tile[load_p, load_tile_size * k_i + load_f] = nl.load(
          k[batch_id, head_id, load_p, j * LARGE_TILE_SZ + load_tile_size * k_i + load_f]
        )
      for v_i in nl.affine_range(LARGE_TILE_SZ // load_tile_size):
        load_p, load_f = nl.mgrid[0:B_D_SIZE, 0:B_P_SIZE]
        store_p, store_f = nl.mgrid[0:B_P_SIZE, 0:B_D_SIZE]
        loaded = nl.load(v[batch_id, head_id, load_p, j * LARGE_TILE_SZ + B_P_SIZE * v_i + load_f], dtype=kernel_dtype)
        cur_v_tile[v_i, store_p, store_f] = nisa.sb_transpose(loaded)

      for i in nl.affine_range(n_tile_q):
        i_f_128 = nl.arange(B_P_SIZE)[None, :]
        i_f_d = nl.arange(B_D_SIZE)[None, :]
        i_p_d = nl.arange(B_D_SIZE)[:,None]
        q_tile = nl.ndarray((B_D_SIZE, B_P_SIZE),dtype=kernel_dtype)
        q_tile[i_p_d, i_f_128] = nl.load(q[batch_id, head_id * q_h_per_k_h + i_q_h, i_p_d, i * B_P_SIZE + i_f_128], dtype=kernel_dtype) \
                                  * softmax_scale # load (d, 128) tile in SBUF
        _flash_attention_core(q_local_tile=q_tile, k=cur_k_tile, v=cur_v_tile,
                              q_h_per_k_h=q_h_per_k_h,
                              o_buffer=o_buffer[i_q_h, i], l_buffer=l_buffer[i_q_h, i], m_buffer=m_buffer[i_q_h, i],
                              batch_id=batch_id, head_id=head_id,
                              gqa_head_idx=i_q_h, q_tile_idx=rank_id * n_tile_q + i,
                              global_k_large_tile_idx=rank_id * num_large_k_tile + j,
                              local_k_large_tile_idx=j,
                              olm_buffer_idx=j,
                              kernel_dtype=kernel_dtype, acc_type=acc_type,
                              flash_config=config, use_causal_mask=use_causal_mask,
                              initialize=False,
                              B_P_SIZE=B_P_SIZE, B_F_SIZE=B_F_SIZE, B_D_SIZE=B_D_SIZE,
                              dropout_p=dropout_p, dropout_p_tensor=dropout_p_tensor, seed_tensor=seed_local)

  # Swap the buffers
  nisa._tiled_offloaded_memcpy(dst=send_k_buf[kv_buf_ix, kv_buf_iy],
                               src=recv_k_buf[kv_buf_ix, kv_buf_iy],)
  nisa._tiled_offloaded_memcpy(dst=send_v_buf[kv_buf_ix, kv_buf_iy],
                               src=recv_v_buf[kv_buf_ix, kv_buf_iy],)
  # Keep receiving the K and V chunks from the left neighbor
  # TODO: Use sequential_range
  for ring_step in nl.static_range(1, num_workers):
    ring_rank_id = ring_order[ring_step]
    nccl.collective_permute(src=send_k_buf[kv_buf_ix, kv_buf_iy],
                            dst=recv_k_buf[kv_buf_ix, kv_buf_iy],
                            replica_groups=replica_groups)
    nccl.collective_permute(src=send_v_buf[kv_buf_ix, kv_buf_iy],
                            dst=recv_v_buf[kv_buf_ix, kv_buf_iy],
                            replica_groups=replica_groups)

    for i_q_h in nl.affine_range(q_h_per_k_h):
      for j in nl.sequential_range(num_large_k_tile):
        cur_k_tile = nl.ndarray((par_dim(B_D_SIZE), LARGE_TILE_SZ), dtype=kernel_dtype)
        cur_v_tile = nl.ndarray((LARGE_TILE_SZ // B_P_SIZE, par_dim(B_P_SIZE), B_D_SIZE), dtype=kernel_dtype)
        load_tile_size = B_P_SIZE
        for k_i in nl.affine_range(LARGE_TILE_SZ // load_tile_size):
          load_p, load_f = nl.mgrid[0:B_D_SIZE, 0:load_tile_size]
          cur_k_tile[load_p, load_tile_size * k_i + load_f] = nl.load(
            send_k_buf[load_p, j * LARGE_TILE_SZ + load_tile_size * k_i + load_f]
          )
        for v_i in nl.affine_range(LARGE_TILE_SZ // load_tile_size):
          load_p, load_f = nl.mgrid[0:B_D_SIZE, 0:B_P_SIZE]
          loaded = nl.load(send_v_buf[load_p, j * LARGE_TILE_SZ + B_P_SIZE * v_i + load_f], dtype=kernel_dtype)
          store_p, store_f = nl.mgrid[0:B_P_SIZE, 0:B_D_SIZE]
          cur_v_tile[v_i, store_p, store_f] = nisa.sb_transpose(loaded)

        for i in nl.affine_range(n_tile_q):
          i_f_128 = nl.arange(B_P_SIZE)[None, :]
          i_f_d = nl.arange(B_D_SIZE)[None, :]
          i_p_d = nl.arange(B_D_SIZE)[:,None]
          q_tile = nl.ndarray((B_D_SIZE, B_P_SIZE),dtype=kernel_dtype)
          q_tile[i_p_d, i_f_128] = nl.load(q[batch_id, head_id * q_h_per_k_h + i_q_h, i_p_d, i * B_P_SIZE + i_f_128], dtype=kernel_dtype) \
                                    * softmax_scale # load (d, 128) tile in SBUF
          _flash_attention_core(q_local_tile=q_tile, k=cur_k_tile, v=cur_v_tile,
                                q_h_per_k_h=q_h_per_k_h,
                                o_buffer=o_buffer[i_q_h, i], l_buffer=l_buffer[i_q_h, i], m_buffer=m_buffer[i_q_h, i],
                                batch_id=batch_id, head_id=head_id,
                                gqa_head_idx=i_q_h, q_tile_idx=rank_id * n_tile_q + i,
                                global_k_large_tile_idx=ring_rank_id * num_large_k_tile + j,
                                local_k_large_tile_idx=j,
                                olm_buffer_idx=ring_step * num_large_k_tile + j,
                                kernel_dtype=kernel_dtype, acc_type=acc_type,
                                flash_config=config, use_causal_mask=use_causal_mask,
                                initialize=False,
                                B_P_SIZE=B_P_SIZE, B_F_SIZE=B_F_SIZE, B_D_SIZE=B_D_SIZE,
                                dropout_p=dropout_p, dropout_p_tensor=dropout_p_tensor, seed_tensor=seed_local)
    
    # Swap the buffer
    nisa._tiled_offloaded_memcpy(dst=send_k_buf[kv_buf_ix, kv_buf_iy],
                                 src=recv_k_buf[kv_buf_ix, kv_buf_iy])
    nisa._tiled_offloaded_memcpy(dst=send_v_buf[kv_buf_ix, kv_buf_iy],
                                 src=recv_v_buf[kv_buf_ix, kv_buf_iy])

  for i_q_h in nl.affine_range(q_h_per_k_h):
    for i in nl.affine_range(n_tile_q):
      # -------- write output to buffer on HBM ------------ #
      out = nl.ndarray((par_dim(B_P_SIZE), B_D_SIZE), dtype=kernel_dtype)
      out[i_q_p, i_f_d] = nl.multiply(o_buffer[i_q_h, i, num_workers * num_large_k_tile - 1, i_q_p, i_f_d], 
                                      nl.exp(m_buffer[i_q_h, i, num_workers * num_large_k_tile - 1, i_q_p, i_0_f] - \
                                             l_buffer[i_q_h, i, num_workers * num_large_k_tile - 1, i_q_p, i_0_f]),
                                      dtype=kernel_dtype)

      nl.store(o[batch_id, head_id * q_h_per_k_h + i_q_h, i * B_P_SIZE + i_q_p, i_f_d], out[i_q_p, i_f_d])
      if not inference:
        lse_local = nl.zeros((par_dim(B_P_SIZE), 1), dtype=acc_type)
        lse_local[i_q_p, i_0_f] = nl.copy(l_buffer[i_q_h, i, num_workers * num_large_k_tile - 1, i_q_p, i_0_f], dtype=acc_type)
        nl.store(lse[batch_id, head_id * q_h_per_k_h + i_q_h, i_q_p, i + i_0_f], lse_local[i_q_p, i_0_f])




def flash_attention(query, key, value):
    out, _ = _mha_forward(query, key, value)
    return out

#@jax.jit
def _mha_forward(query, key, value):
    # Get the batch size, sequence lengths, number of heads, and hidden dimension
    batch_size, q_seq_len, num_heads, d_model = query.shape
    _, kv_seq_len, _, _ = key.shape
    
    # Transpose the query, key, and value tensors
    q = query.transpose(0, 2, 3, 1)  # [batch_size, num_heads, d_model, q_seq_len]
    k = key.transpose(0, 2, 3, 1)  # [batch_size, num_heads, d_model, kv_seq_len]
    v = value.transpose(0, 2, 1, 3)  # [batch_size, num_heads, kv_seq_len, d_model]
    
    # Create the output buffer
    attn_output_shape = jax.ShapeDtypeStruct((batch_size, num_heads, q_seq_len, d_model), dtype=query.dtype)
    lse_shape = jax.ShapeDtypeStruct((batch_size, num_heads, nl.tile_size.pmax, q_seq_len // nl.tile_size.pmax), dtype=jnp.float32)
    attn_output = jnp.zeros((batch_size, num_heads, q_seq_len, d_model), dtype=query.dtype)
    lse = jnp.zeros((batch_size, num_heads, nl.tile_size.pmax, q_seq_len // nl.tile_size.pmax), dtype=jnp.float32)
    seed = jnp.array([1])
    # Call the NKI kernel using nki_call
    attn_output, lse = nki_call(
        flash_fwd,
        q, k, v, seed, attn_output, lse,
        out_shape=(attn_output_shape, lse_shape),
        grid=(batch_size, num_heads)
    )
    # Transpose the output back to the original shape
    attn_output = attn_output.transpose(0, 2, 1, 3)  # [batch_size, q_seq_len, num_heads, d_model]
    
    return attn_output, (lse, attn_output, q, k, v)

#@jax.jit
def _mha_backward(res, d_attn_output):
    lse, o, q, k, v = res
    #print(f'Q shape {q.shape}')
    #print(f'K shape {k.shape}')
    #print(f'V shape {v.shape}')
    batch_size, q_seq_len, num_heads, d_model = q.shape
    _, kv_seq_len, _, _ = k.shape

    # Transpose the input tensors
    o = o.transpose(0, 2, 3, 1)
    dy = d_attn_output.transpose(0, 2, 3, 1)

    # Transpose v tensor
    v = jnp.transpose(v, axes=(0, 1, 3, 2))
    # Create the output buffer shapes
    d_query_shape = jax.ShapeDtypeStruct(q.shape, q.dtype)
    d_key_shape = jax.ShapeDtypeStruct(k.shape, k.dtype)
    d_value_shape = jax.ShapeDtypeStruct(v.shape, v.dtype)
    d_query = jnp.zeros(q.shape, dtype=q.dtype)
    d_key = jnp.zeros(k.shape, dtype=q.dtype)
    d_value = jnp.zeros(v.shape, dtype=q.dtype)
    seed = jnp.array([1])

    # Call the NKI kernel using nki_call
    d_query, d_key, d_value = nki_call(
        flash_attn_bwd,
        q, k, v, o, dy, lse, seed, d_query, d_key, d_value,
        out_shape=[d_query_shape, d_key_shape, d_value_shape],
        grid=(batch_size, num_heads)
    )
    #print(f'D Query: {d_query.shape}')
    #print(f'D Key: {d_key.shape}')
    #print(f'D Value: {d_value.shape}')

    # Batch seq_len heads, head_dim
    # Transpose the gradients back to the original shape
    d_query = d_query.transpose(0, 3, 1, 2)
    d_key = d_key.transpose(0, 3, 1, 2)
    d_value = d_value.transpose(0, 3, 1, 2)

    return d_query, d_key, d_value

flash_attention = jax.custom_vjp(flash_attention)
flash_attention.defvjp(_mha_forward, _mha_backward)