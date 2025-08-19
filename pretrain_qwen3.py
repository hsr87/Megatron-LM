#!/usr/bin/env python
"""Pretrain Qwen3 8B model."""

import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import timedelta

from megatron.core import mpu, tensor_parallel
from megatron.core.enums import ModelType
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_with_transformer_engine_spec
from megatron.core.transformer.spec_utils import import_module
from megatron.training import get_args, get_tokenizer, print_rank_0
from megatron.training import global_vars
from megatron.training.checkpointing import load_checkpoint, save_checkpoint
from megatron.training.initialize import initialize_megatron
from megatron.training.training import pretrain
from megatron.legacy.model import GPTModel as LegacyGPTModel
from megatron.core.transformer.module import MegatronModule
from megatron.core.models.common.embeddings.language_model_embedding import LanguageModelEmbedding
from megatron.core.models.common.embeddings.rotary_pos_embedding import RotaryEmbedding
from megatron.core.datasets.blended_megatron_dataset_builder import BlendedMegatronDatasetBuilder
from megatron.core.datasets.gpt_dataset import GPTDatasetConfig, MockGPTDataset

def model_provider(pre_process=True, post_process=True):
    """Build the Qwen3 8B model.
    
    Qwen3 8B architecture details:
    - Hidden size: 4096
    - Number of layers: 64
    - Number of attention heads: 32
    - Number of key-value heads: 8 (GQA)
    - FFN hidden size: 14336
    - Max position embeddings: 131072
    - Vocab size: 152064
    - RoPE base: 1000000
    """
    args = get_args()
    
    # Override with Qwen3 8B specific parameters
    args.hidden_size = 4096
    args.num_layers = 64
    args.num_attention_heads = 32
    args.num_key_value_heads = 8  # GQA
    args.ffn_hidden_size = 14336
    args.max_position_embeddings = 131072
    args.vocab_size = 152064
    args.rotary_base = 1000000
    args.norm_epsilon = 1e-6
    args.use_rotary_position_embeddings = True
    args.rotary_interleaved = False
    args.swiglu = True
    args.normalization = 'RMSNorm'
    args.add_bias_linear = False
    args.untie_embeddings_and_output_weights = True
    
    print_rank_0('Building Qwen3 8B model ...')
    
    config = core_gpt_config_from_args(args)
    
    if args.use_legacy_models:
        model = LegacyGPTModel(
            config,
            num_tokentypes=0,
            parallel_output=True,
            pre_process=pre_process,
            post_process=post_process
        )
    else:
        transformer_layer_spec = get_gpt_layer_with_transformer_engine_spec(
            args.num_experts, args.moe_grouped_gemm,
            args.qk_layernorm, args.multi_query_attention
        )
        
        model = GPTModel(
            config=config,
            transformer_layer_spec=transformer_layer_spec,
            vocab_size=args.padded_vocab_size,
            max_sequence_length=args.max_position_embeddings,
            pre_process=pre_process,
            post_process=post_process,
            fp16_lm_cross_entropy=args.fp16_lm_cross_entropy,
            parallel_output=True,
            share_embeddings_and_output_weights=not args.untie_embeddings_and_output_weights,
            position_embedding_type=args.position_embedding_type,
            rotary_percent=args.rotary_percent,
            rotary_base=args.rotary_base,
            seq_len_interpolation_factor=args.rotary_seq_len_interpolation_factor
        )
    
    return model

def core_gpt_config_from_args(args):
    """Create core GPT config from arguments."""
    from megatron.core.transformer import TransformerConfig
    
    # Qwen3 uses SwiGLU activation
    if args.swiglu:
        args.ffn_hidden_size = args.ffn_hidden_size * 2 // 3
        activation = 'swiglu'
    else:
        activation = 'gelu'
    
    # Configure normalization for Qwen3 (RMSNorm)
    if args.normalization == "RMSNorm":
        normalization = 'RMSNorm'
    else:
        normalization = 'LayerNorm'
    
    config = TransformerConfig(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        ffn_hidden_size=args.ffn_hidden_size,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads if args.num_key_value_heads else args.num_attention_heads,
        kv_channels=args.kv_channels,
        init_method_std=args.init_method_std,
        hidden_dropout=args.hidden_dropout,
        attention_dropout=args.attention_dropout,
        ffn_dropout=args.ffn_dropout,
        layernorm_epsilon=args.norm_epsilon,
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        virtual_pipeline_model_parallel_size=args.virtual_pipeline_model_parallel_size,
        sequence_parallel=args.sequence_parallel,
        activation_func=activation,
        normalization=normalization,
        add_bias_linear=args.add_bias_linear,
        bias_activation_fusion=args.bias_activation_fusion,
        bias_dropout_fusion=args.bias_dropout_fusion,
        apply_layernorm_1p=args.apply_layernorm_1p,
        apply_residual_connection_post_layernorm=args.apply_residual_connection_post_layernorm,
        fp32_residual_connection=args.fp32_residual_connection,
        bf16=args.bf16,
        fp16=args.fp16,
        fp8=args.fp8,
        params_dtype=args.params_dtype,
        timers=args.timers,
        recompute_granularity=args.recompute_granularity,
        recompute_method=args.recompute_method,
        recompute_num_layers=args.recompute_num_layers,
        distribute_saved_activations=args.distribute_saved_activations,
    )
    
    return config

def get_batch(data_iterator):
    """Generate a batch."""
    args = get_args()
    tokenizer = get_tokenizer()
    
    # Items and their type.
    keys = ['text']
    datatype = torch.int64
    
    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = tensor_parallel.broadcast_data(keys, data, datatype)
    
    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()
    
    # Get the masks and position ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss
    )
    
    return tokens, labels, loss_mask, attention_mask, position_ids

def get_ltor_masks_and_position_ids(tokens, eod_token, reset_position_ids,
                                   reset_attention_mask, eod_mask_loss):
    """Build left-to-right masks and position ids."""
    batch_size, seq_length = tokens.size()
    
    # Position ids
    position_ids = torch.arange(seq_length, dtype=torch.long, device=tokens.device)
    position_ids = position_ids.unsqueeze(0).expand_as(tokens)
    
    # Loss mask
    loss_mask = torch.ones(tokens.size(), dtype=torch.float, device=tokens.device)
    
    # Attention mask (causal)
    attention_mask = torch.tril(torch.ones(
        (batch_size, seq_length, seq_length), device=tokens.device
    )).view(batch_size, 1, seq_length, seq_length)
    
    # Convert attention mask to bool
    attention_mask = (attention_mask < 0.5)
    
    return attention_mask, loss_mask, position_ids

def loss_func(loss_mask, output_tensor):
    """Loss function for Qwen3."""
    args = get_args()
    
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    
    if args.context_parallel_size > 1:
        loss = torch.cat([torch.sum(losses.view(-1) * loss_mask).view(1), 
                         loss_mask.sum().view(1)])
        torch.distributed.all_reduce(loss, group=mpu.get_context_parallel_group())
        loss = loss[0] / loss[1]
    else:
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
    
    # Check for NaN
    if args.check_for_nan_in_loss_and_grad:
        global_vars.update_nan_in_loss(loss)
    
    # Reduce loss for logging
    averaged_loss = average_losses_across_data_parallel_group([loss])
    
    return loss, {'lm loss': averaged_loss[0]}

def average_losses_across_data_parallel_group(losses):
    """Average losses across data parallel group."""
    args = get_args()
    averaged_losses = []
    for loss in losses:
        if args.sequence_parallel:
            torch.distributed.all_reduce(loss, 
                                        group=mpu.get_sequence_parallel_group())
        else:
            torch.distributed.all_reduce(loss,
                                        group=mpu.get_data_parallel_group())
        averaged_losses.append(loss / mpu.get_data_parallel_world_size())
    
    return averaged_losses

def forward_step(data_iterator, model):
    """Forward training step."""
    args = get_args()
    
    # Get batch
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(data_iterator)
    
    # Forward pass
    output_tensor = model(tokens, position_ids, attention_mask,
                         labels=labels,
                         inference_params=None)
    
    return output_tensor, partial(loss_func, loss_mask)

def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets for Qwen3."""
    args = get_args()
    
    print_rank_0('> building train, validation, and test datasets for Qwen3 ...')
    
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        data_prefix=args.data_path,
        data_impl=args.data_impl,
        splits_string=args.split,
        train_valid_test_num_samples=train_val_test_num_samples,
        seq_length=args.seq_length,
        seed=args.seed,
        skip_warmup=(not args.mmap_warmup),
        train_data_prefix=args.train_data_path,
        valid_data_prefix=args.valid_data_path,
        test_data_prefix=args.test_data_path,
        data_cache_path=args.data_cache_path
    )
    
    print_rank_0("> finished creating Qwen3 datasets ...")
    
    return train_ds, valid_ds, test_ds

def build_train_valid_test_datasets(data_prefix, data_impl, splits_string,
                                   train_valid_test_num_samples,
                                   seq_length, seed, skip_warmup,
                                   train_data_prefix=None,
                                   valid_data_prefix=None,
                                   test_data_prefix=None,
                                   data_cache_path=None):
    """Build datasets."""
    # For now, use MockGPTDataset for demonstration
    # In production, use actual dataset loading
    config = GPTDatasetConfig(
        random_seed=seed,
        sequence_length=seq_length,
        blend_per_split=None,
        split=splits_string,
        path_to_cache=data_cache_path,
        mock=True,
        tokenizer=get_tokenizer()
    )
    
    print_rank_0(" > building mock dataset for Qwen3 ...")
    
    train_ds = MockGPTDataset(train_valid_test_num_samples[0], config)
    valid_ds = MockGPTDataset(train_valid_test_num_samples[1], config)
    test_ds = MockGPTDataset(train_valid_test_num_samples[2], config)
    
    return train_ds, valid_ds, test_ds

def add_qwen3_args(parser):
    """Add Qwen3-specific arguments."""
    group = parser.add_argument_group(title='Qwen3')
    
    group.add_argument('--num-key-value-heads', type=int, default=None,
                      help='Number of key-value heads for GQA. Default is None (MHA).')
    group.add_argument('--rotary-base', type=float, default=10000,
                      help='Base for rotary position embeddings')
    group.add_argument('--normalization', type=str, default='LayerNorm',
                      choices=['LayerNorm', 'RMSNorm'],
                      help='Normalization type')
    group.add_argument('--swiglu', action='store_true',
                      help='Use SwiGLU activation')
    group.add_argument('--untie-embeddings-and-output-weights', action='store_true',
                      help='Untie embeddings and output weights')
    
    return parser

def extra_args_provider(parser):
    """Provide extra arguments for Qwen3."""
    parser = add_qwen3_args(parser)
    return parser

if __name__ == '__main__':
    # Temporary workaround for import issues
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from functools import partial
    
    # Pretrain Qwen3 8B
    pretrain(train_valid_test_datasets_provider,
             model_provider,
             ModelType.encoder_or_decoder,
             forward_step,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'},
             extra_args_provider=extra_args_provider)