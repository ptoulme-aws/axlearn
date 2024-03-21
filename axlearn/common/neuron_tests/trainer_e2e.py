import contextlib

# pylint: disable=too-many-lines,duplicate-code,no-self-use

import jax
import pytest
import numpy as np
import optax
from jax import numpy as jnp
from jax.sharding import PartitionSpec
from jax.sharding import NamedSharding
from axlearn.common import attention, test_utils, utils, causal_lm, optimizers
from axlearn.common.attention import (
    ParallelTransformerLayer,
    TransformerLayer, scaled_hidden_dim, TransformerFeedForwardLayer, MultiheadAttention, FusedQKVLinear, QKVLinear,
    StackedTransformerLayer, RepeatedTransformerLayer, PipelinedTransformerLayer, build_remat_spec,
)
from axlearn.common.base_layer import ParameterSpec, RematSpec
from axlearn.common.causal_lm import residual_initializer_cfg, TransformerStackConfig
from axlearn.common.config import config_for_function
from axlearn.common.decoder import Decoder, LmHead
from axlearn.common.embedding import TransformerTextEmbeddings
from axlearn.common.layers import RMSNorm, set_bias_recursively
from axlearn.common.learner import Learner
from axlearn.common.module import functional as F, InvocationContext, new_output_collection, set_current_context
from axlearn.common.optimizer_base import NestedOptParam, OptParam
from axlearn.common.optimizers import AddDecayedWeightsState
from axlearn.common.test_utils import NeuronTestCase, assert_allclose, dummy_segments_positions
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

from axlearn.common.utils import Tensor, VDict, NestedTensor, TensorSpec, with_sharding_constraint
import os


def llama_trainer():
    TP_DEGREE = 8
    DP_DEGREE = (int(os.getenv('SLURM_JOB_NUM_NODES'))*32)//TP_DEGREE
    mesh = jax.sharding.Mesh(np.array(jax.devices()).reshape(DP_DEGREE, TP_DEGREE)[:, None, None, None, :],
                                axis_names=("data", "seq", "expert", "fsdp", "model"),)

    with mesh:
        model_dim = 4096
        num_heads = 32
        vocab_size = 32000
        stacked_layer = StackedTransformerLayer.default_config()
        decoder_cfg = llama_decoder_config(
            stack_cfg=stacked_layer,
            num_layers=4,
            hidden_dim=model_dim,
            num_heads=num_heads,
            vocab_size=vocab_size,
            activation_function="nn.gelu",
            layer_norm_epsilon=0.1,
            dropout_rate=0.0,
        )
        model_cfg = causal_lm.Model.default_config().set(decoder=decoder_cfg, name="llama")
        #print(model_cfg)
        set_model_shard_weights_config(
            model_cfg,
            batch_axis_names='data',
            fsdp_axis_names='data',
            tp_axis_names='model',
            seq_axis_names='model',
        )

def llama_decoder_config(
        stack_cfg: TransformerStackConfig,
        num_layers: int,
        hidden_dim: int,
        num_heads: int,
        vocab_size: int,
        activation_function: str = "nn.relu",
        layer_norm_epsilon: float = 1e-08,
        dropout_rate: float = 0.0,
        layer_remat: Optional[RematSpec] = None,
) -> Decoder.Config:
    """Build a decoder transformer config in the style of GPT.

    Reference: https://github.com/openai/gpt-2.

    Args:
        stack_cfg: A config of StackedTransformerLayer, RepeatedTransformerLayer, or
            PipelinedTransformerLayer.
        num_layers: Number of transformer decoder layers.
        hidden_dim: Dimension of embeddings and input/output of each transformer layer.
        num_heads: Number of attention heads per transformer layer.
        vocab_size: Size of vocabulary.
        max_position_embeddings: Number of positional embeddings.
        activation_function: Type of activation function.
        layer_norm_epsilon: Epsilon for layer normalization. Defaults to LayerNorm.config.eps.
        dropout_rate: Dropout rate applied throughout model, including output_dropout.
        layer_remat: If not None, use as transformer.layer.remat_spec.

    Returns:
        A Decoder config.
    """
    stack_cfg = stack_cfg.clone()

    assert stack_cfg.klass in [
        StackedTransformerLayer,
        RepeatedTransformerLayer,
        PipelinedTransformerLayer,
    ]

    cfg = TransformerLayer.default_config()
    cfg.dtype = jnp.bfloat16
    #cfg.dtype = jnp.float32
    cfg.feed_forward.set(hidden_dim=scaled_hidden_dim(4))
    cfg.self_attention.attention.set(num_heads=num_heads)
    cfg.self_attention.attention.input_linear = FusedQKVLinear.default_config()
    cfg.self_attention.norm = RMSNorm.default_config()
    cfg.feed_forward.norm = RMSNorm.default_config()
    set_bias_recursively(cfg, bias=False)

    transformer_cls = stack_cfg.set(num_layers=num_layers, layer=cfg)
    decoder = Decoder.default_config().set(
        transformer=transformer_cls,
        dim=hidden_dim,
        vocab_size=vocab_size,
        emb=TransformerTextEmbeddings.default_config().set(pos_emb=None).set(dtype=jnp.bfloat16), #bfloat16
        output_norm=RMSNorm.default_config().set(eps=layer_norm_epsilon),
        dropout_rate=dropout_rate,
        lm_head=LmHead.default_config().set(dtype=jnp.bfloat16)  #bfloat16
    )
    return decoder

def set_model_shard_weights_config(
        cfg: Union[TransformerLayer.Config, Sequence[TransformerLayer.Config]],
        *,
        batch_axis_names: Union[str, Sequence[str]] = ("data", "fsdp"),
        fsdp_axis_names: Union[str, Sequence[str]] = "fsdp",
        tp_axis_names: Union[str, Sequence[str]] = "model",
        seq_axis_names: Union[str, Sequence[str]] = "seq",
):
    """Sets `cfg` to shard FFN and attention weights over both fsdp and tp axes.

    Args:
        cfg: (A sequence of) Transformer layer config to apply sharding spec to.
        batch_axis_names: Axis name(s) over which we shard the batch dimension of output tensors.
        fsdp_axis_names: Axis name(s) over which we shard fully-sharded-data-parallel tensors.
        tp_axis_names: Axis name(s) over which we shard tensor-parallel tensors.
        seq_axis_names: Axis name(s) over which we shard sequence-parallel tensors.
    """

    # pytype: disable=attribute-error
    def set_attn_partition_specs(attn_layer: MultiheadAttention.Config):
        # Shard weights.
        input_linear_cfg = attn_layer.input_linear
        if hasattr(input_linear_cfg, "input_linear"):
            input_linear_cfg = input_linear_cfg.input_linear
        input_linear_cfg.layer.param_partition_spec = (None, fsdp_axis_names, tp_axis_names, None)
        # ptoulme bug - when FusedQKV is enabled it has a shape (3, hidden, num_heads, head_dimension) dimension so add a (None to account for this
        attn_layer.output_linear.param_partition_spec = (fsdp_axis_names, tp_axis_names, None)

    def set_ffn_partition_specs(ff_layer: TransformerFeedForwardLayer.Config):
        # Shard weights.
        ff_layer.linear1.param_partition_spec = (fsdp_axis_names, tp_axis_names)
        ff_layer.linear2.param_partition_spec = (tp_axis_names, fsdp_axis_names)
        # Encourage the right activation sharding.
        ff_layer.linear1.output_partition_spec = (batch_axis_names, None, tp_axis_names)
        ff_layer.linear2.output_partition_spec = (batch_axis_names, None, None)

    #if not isinstance(cfg, Sequence):
     #   cfg = [cfg]
    #print(cfg.decoder)
    cfg.decoder.emb.token_emb.param_partition_spec = (tp_axis_names, fsdp_axis_names) # shard hidden
    cfg.decoder.lm_head.param_partition_spec = (tp_axis_names, fsdp_axis_names) # shard vocab
    for layer_cfg in [cfg.decoder.transformer.layer]: # shard the sole layer and its used for all other layers
        layer_cfg.remat_spec = build_remat_spec(cfg.decoder.transformer) # activation checkpointing
        set_attn_partition_specs(layer_cfg.self_attention.attention)
        if layer_cfg.cross_attention is not None:
            set_attn_partition_specs(layer_cfg.cross_attention.attention)
        if isinstance(layer_cfg.feed_forward, TransformerFeedForwardLayer.Config):
            set_ffn_partition_specs(layer_cfg.feed_forward)