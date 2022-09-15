# This file is an all-encompassing module which contains all logic for
# constructing and loading the GLM model. Logic for inference can be found
# in the inference_glm.py file in the examples folder (in the SwissArmyTransformer
# repository).
#
# This version of the file is based off of the version within SwissArmyTransformer.
#

from dataclasses import dataclass, field
from typing import Optional
from functools import partial
import argparse
import os
import copy
import json
import logging
import sys
import math
import random
import requests

from filelock import FileLock
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

from metaseq.dataclass import MetaseqDataclass
from metaseq.dataclass.utils import gen_parser_from_dataclass
from metaseq.models import register_model

from megatron import mpu
from megatron.mpu.initialize import get_tensor_model_parallel_world_size, get_tensor_model_parallel_rank, get_data_parallel_rank
from megatron.mpu.layers import ColumnParallelLinear, RowParallelLinear, VocabParallelEmbedding
from megatron.mpu.mappings import gather_from_tensor_model_parallel_region, copy_to_tensor_model_parallel_region
from megatron.mpu.utils import divide, split_tensor_along_last_dim

#from deepspeed.runtime.activation_checkpointing.checkpointing import checkpoint

# Import fused layer norm with fallback to regular layer norm
try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm
    class LayerNorm(FusedLayerNorm):
        def __init__(self, *args, pb_relax=False, **kwargs):
            super().__init__(*args, **kwargs)
            self.pb_relax = pb_relax

        def forward(self, x):
            if not self.pb_relax:
                return super().forward(x)
            return super().forward(x / (x.abs().max().detach() / 8))
except ModuleNotFoundError:
    print('Please install apex to use fused_layer_norm, fall back to torch.nn.LayerNorm')
    from  torch.nn import LayerNorm


logger = logging.getLogger(__name__)


def unscaled_init_method(sigma):
    """Init method based on N(0, sigma)."""
    def init_(tensor, **kwargs):
        return torch.nn.init.normal_(tensor, mean=0.0, std=sigma)

    return init_

def scaled_init_method(sigma, num_layers):
    """Init method based on N(0, sigma/sqrt(2*num_layers)."""
    std = sigma / math.sqrt(2.0 * num_layers)
    def init_(tensor, **kwargs):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)

    return init_


@torch.jit.script
def gelu_impl(x):
     """OpenAI's gelu implementation."""
     return 0.5 * x * (1.0 + torch.tanh(0.7978845608028654 * x *
                                        (1.0 + 0.044715 * x * x)))

def gelu(x): 
    return gelu_impl(x)


def standard_attention(query_layer, key_layer, value_layer, attention_mask,
                       attention_dropout=None, log_attention_weights=None, scaling_attention_score=True, **kwargs):
    # We disable the PB-relax-Attention and only changes the order of computation, because it is enough for most of training. 
    # The implementation in the paper can be done very easily, if you really need it to train very deep transformers. 

    if scaling_attention_score:
        query_layer = query_layer / math.sqrt(query_layer.shape[-1])
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
    if log_attention_weights is not None:
        attention_scores += log_attention_weights

    if not (attention_mask.shape[-2] == 1 and (attention_mask > 0).all()):
        # if auto-regressive, skip
        attention_scores = torch.mul(attention_scores, attention_mask) - \
                           10000.0 * (1.0 - attention_mask)

    attention_probs = F.softmax(attention_scores, dim=-1)

    if attention_dropout is not None:
        if mpu.get_cuda_rng_tracker is not None:
            with mpu.get_cuda_rng_tracker().fork():
                attention_probs = attention_dropout(attention_probs)
        else:
            attention_probs = attention_dropout(attention_probs)

    context_layer = torch.matmul(attention_probs, value_layer)
    return context_layer

def attention_forward_default(self, hidden_states, mask, **kw_args):
    self = self.transformer.layers[kw_args['layer_id']].attention
    attention_fn = standard_attention
    if 'attention_fn' in self.hooks:
        attention_fn = self.hooks['attention_fn']

    mixed_raw_layer = self.query_key_value(hidden_states)
    (mixed_query_layer,
        mixed_key_layer,
        mixed_value_layer) = split_tensor_along_last_dim(mixed_raw_layer, 3)

    dropout_fn = self.attention_dropout if self.training else None

    query_layer = self._transpose_for_scores(mixed_query_layer)
    key_layer = self._transpose_for_scores(mixed_key_layer)
    value_layer = self._transpose_for_scores(mixed_value_layer)

    context_layer = attention_fn(query_layer, key_layer, value_layer, mask, dropout_fn, **kw_args)

    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
    context_layer = context_layer.view(*new_context_layer_shape)
    output = self.dense(context_layer)

    if self.training:
        output = self.output_dropout(output)
    return output

def cross_attention_forward_default(self, hidden_states, cross_attention_mask, encoder_outputs, **kw_args):
    self = self.transformer.layers[kw_args['layer_id']].cross_attention
    attention_fn = standard_attention
    if 'attention_fn' in self.hooks:
        attention_fn = self.hooks['attention_fn']

    mixed_query_layer = self.query(hidden_states)
    mixed_x_layer = self.key_value(encoder_outputs)
    (mixed_key_layer, mixed_value_layer) = split_tensor_along_last_dim(mixed_x_layer, 2)

    dropout_fn = self.attention_dropout if self.training else None
    # Reshape and transpose [b, np, s, hn]
    query_layer = self._transpose_for_scores(mixed_query_layer)
    key_layer = self._transpose_for_scores(mixed_key_layer)
    value_layer = self._transpose_for_scores(mixed_value_layer)

    context_layer = attention_fn(query_layer, key_layer, value_layer, cross_attention_mask, dropout_fn, **kw_args)
    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size_per_partition,)
    # [b, s, hp]
    context_layer = context_layer.view(*new_context_layer_shape)

    # Output. [b, s, h]
    output = self.dense(context_layer)
    if self.training:
        output = self.output_dropout(output)
    return output

def mlp_forward_default(self, hidden_states, **kw_args):
    self = self.transformer.layers[kw_args['layer_id']].mlp
    intermediate_parallel = self.dense_h_to_4h(hidden_states)
    intermediate_parallel = self.activation_func(intermediate_parallel)
    output = self.dense_4h_to_h(intermediate_parallel)
    return output

def word_embedding_forward_default(self, input_ids, output_cross_layer, **kw_args):
    return self.transformer.word_embeddings(input_ids)

def position_embedding_forward_default(self, position_ids, output_cross_layer, **kw_args):
    return self.transformer.position_embeddings(position_ids)

def final_forward_default(self, logits, **kw_args):
    return F.linear(logits, self.transformer.word_embeddings.weight)

def layer_forward_default(self, hidden_states, mask, *args, **kw_args):
    '''
        hidden_states: [batch, seq_len, hidden_size]
        mask: [(1, 1), seq_len, seq_len]
    '''
    self = self.transformer.layers[kw_args['layer_id']]
    # Layer norm at the begining of the transformer layer.
    attention_input = self.input_layernorm(hidden_states)
    # Self attention.
    attention_output = self.attention(attention_input, mask, **kw_args)

    # Third LayerNorm
    if self.layernorm_order == 'sandwich':
        attention_output = self.third_layernorm(attention_output)
    
    # Residual connection.
    if self.layernorm_order == 'post':
        hidden_states = attention_input + attention_output
    else:
        hidden_states = hidden_states + attention_output

    
    mlp_input = self.post_attention_layernorm(hidden_states)

    if self.is_decoder:
        encoder_outputs = kw_args['encoder_outputs']
        if encoder_outputs is not None:
            assert 'cross_attention_mask' in kw_args
            # Cross attention
            attention_output = self.cross_attention(mlp_input, **kw_args)
            # Residual connection.
            hidden_states = hidden_states + attention_output
            # Layer norm post the cross attention
            mlp_input = self.post_cross_attention_layernorm(hidden_states)

    # MLP.
    mlp_output = self.mlp(mlp_input, **kw_args)

    # Fourth LayerNorm
    if self.layernorm_order == 'sandwich':
        mlp_output = self.fourth_layernorm(mlp_output)

    # Second residual connection.
    if self.layernorm_order == 'post':
        output = mlp_input + mlp_output
    else:
        output = hidden_states + mlp_output

    return output

HOOKS_DEFAULT = {
    'attention_fn': standard_attention,
    'attention_forward': attention_forward_default,
    'cross_attention_forward': cross_attention_forward_default,
    'mlp_forward': mlp_forward_default,
    'word_embedding_forward': word_embedding_forward_default,
    'position_embedding_forward': position_embedding_forward_default,
    'final_forward': final_forward_default,
    'layer_forward': layer_forward_default
}


def non_conflict(func):
    func.non_conflict = True
    return func

class BaseMixin(torch.nn.Module):
    non_conflict = non_conflict
    def __init__(self):
        super(BaseMixin, self).__init__()
        # define new params

    def reinit(self, parent_model=None):
        # reload the initial params from previous trained modules
        # you can also get access to other mixins through parent_model.get_mixin().
        pass

    # can define hook-functions here
    # ...

    # If the hook is just a pre- or post- transformation,
    # You can use @non_conflict to mark it,
    # and run `old_impl` to make it compatible with other mixins.
    # Eg., 
    # 
    # @non_conflict
    # def attention_fn(q, k, v, mask, dropout_fn, old_impl=standard_attention, **kw_args):
    #     new_q, new_k, new_v = pre_hack(q, k, v)
    #     attn_result = old_impl(q, k, v, mask, dropout_fn, **kw_args)
    #     attn_result = post_hack(attn_result)
    #     return attn_result


def get_checkpoint_name(checkpoints_path, iteration, release=False, zero=False):
    if release:
        d = 'release'
    else:
        d = '{:d}'.format(iteration)
    if zero:
        dp_rank = get_data_parallel_rank()
        d += '_zero_dp_rank_{}'.format(dp_rank)
    return os.path.join(checkpoints_path, d, 'mp_rank_{:02d}_model_states.pt'.format(get_tensor_model_parallel_rank()))


def get_checkpoint_tracker_filename(checkpoints_path, old_checkpoint=False):
    return os.path.join(checkpoints_path, 'latest')


def get_checkpoint_iteration(load_path):
    # Read the tracker file and set the iteration.
    tracker_filename = get_checkpoint_tracker_filename(load_path)
    if not os.path.isfile(tracker_filename):
        print_rank_0('could not find the metadata file {} '.format(
            tracker_filename))
        raise ValueError('could not find the metadata file {}, please check --load'.format(
            tracker_filename))
        return 0, False, False
    iteration = 0
    release = False
    with open(tracker_filename, 'r') as f:
        metastring = f.read().strip()
        try:
            iteration = int(metastring)
        except ValueError:
            release = metastring == 'release'
            if not release:
                print_rank_0('ERROR: Invalid metadata file {}. Exiting'.format(
                    tracker_filename))
                exit()
    assert iteration > 0 or release, 'error parsing metadata file {}'.format(
        tracker_filename)

    return iteration, release, True


def get_model(cfg, model_cls):
    """Build the model."""

    #print_rank_0(f'building {model_cls.__name__} model ...')
    model = model_cls(cfg)

    logger.info(
        "Number of parameters on model parallel rank {}: {}".format(
            cfg.distributed_training.distributed_rank,
            sum([p.nelement() for p in model.parameters()])))

    if cfg.common.fp16:
        model.half()
    elif cfg.common.bf16:
        model.bfloat16()
    model.cuda(torch.cuda.current_device())

    return model

def load_checkpoint(model, cfg, load_path=None, prefix=''):
    """Load a model checkpoint."""
    if load_path is None:
        load_path = cfg.load

    iteration, release, success = get_checkpoint_iteration(load_path)
    if not success:
        return 0
    
    checkpoint_name = get_checkpoint_name(load_path, iteration, release)

    if get_data_parallel_rank() == 0:
        logger.info(f"Global rank {torch.distributed.get_rank()} is "
                    f"loading checkpoint {checkpoint_name}")

    try:
        sd = torch.load(checkpoint_name, map_location='cpu')
    except FileNotFoundError:
        print(f"Model checkpoint under {checkpoint_name} does not exist")

    new_sd = {'module':{}}
    for k in sd:
        if k != 'module':
            new_sd[k] = sd[k]
    for k in sd['module']:
        if k.startswith(prefix):
            new_sd['module'][k[len(prefix):]] = sd['module'][k]
    sd = new_sd
    
    if hasattr(model, 'module'):
        module = model.module
    else: # inference without deepspeed
        module = model


    # TODO: Checkpoint loads, but since GLM-large is a singleton checkpoint and
    #       there are no native resharding utilities, there are size mismatches
    #       during the module.load_state_dict() call goes through. We should
    #       facilitate resharding across ranks if needed.
    # only load module, other hyperparameters are just for recording.
    breakpoint()
    missing_keys, unexpected_keys = module.load_state_dict(sd['module'], strict=False)


    if len(unexpected_keys) > 0:
        print_rank_0(
            f'Will continue but found unexpected_keys! Check whether you are loading correct checkpoints: {unexpected_keys}.')
    if len(missing_keys) > 0:
        raise ValueError(f'Missing keys for inference: {missing_keys}.')

        """
        if args.mode == 'inference':
        else: # new params
            assert all(name.find('mixins')>=0 for name in missing_keys), missing_keys
            assert args.mode == 'finetune'
            # list all mixin names
            mixin_names = []
            for key_name in missing_keys:
                parts = key_name.split('.')
                mixin_name = parts[parts.index('mixins')+1]
                if mixin_name not in mixin_names:
                    mixin_names.append(mixin_name)
            module.reinit(mixin_names) # initialize mixins
        """

    # Do not need this any more, because we create optimizer after load now.
    # if args.mode != 'inference' and args.deepspeed and args.fp16:
    #     model.optimizer.refresh_fp32_params() # restore fp32 weights from module

    # Iterations.
    """
    if args.mode == 'finetune':
        iteration = 0
    elif args.mode == 'pretrain' and not args.no_load_rng: # rng states.
        try:
            random.setstate(sd['random_rng_state'])
            np.random.set_state(sd['np_rng_state'])
            torch.set_rng_state(sd['torch_rng_state'])
            torch.cuda.set_rng_state(sd['cuda_rng_state'])
            mpu.get_cuda_rng_tracker().set_states(sd['rng_tracker_states'])
        except KeyError:
            print_rank_0('Unable to load optimizer from checkpoint {}, exiting. '
                         'Specify --no-load-rng or --finetune to prevent '
                         'attempting to load the random '
                         'state.'.format(checkpoint_name))
            exit()
    """
    #elif args.mode == 'inference':
    module.eval()

    if get_data_parallel_rank() == 0:
        print('  successfully loaded {}'.format(checkpoint_name))
    del sd
    return iteration


def update_args_with_file(cfg, path):
    """
    Open the metadata file included with pretrained SwissArmyTransformer, and
    replace main cfg default args with those found in the metadata file
    """
    with open(path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    # expand relative path
    folder = os.path.dirname(path)
    """
    for k in config:
        # all the relative paths in config are based on the folder
        if k.endswith('_path'): 
            config[k] = os.path.join(folder, config[k])
            if cfg.distributed_training.distributed_rank == 0:
                print(f'> parsing relative path {k} in model_config as {config[k]}.')
    """
    # TODO: This can all be done in one loop
    # Add new keys present in loaded model config
    OmegaConf.set_struct(cfg, False)    # This should only be used when loading configs from ckpts
    for k in config.keys():
        if k not in cfg.model:
            cfg.model.k = config[k]

    # Overwrite duplicate keys in cfg.model with loaded model config
    for k in cfg.model.keys():
        if k in config and cfg.model[k] != config[k]:
            if cfg.distributed_training.distributed_rank == 0:
                logger.info(f"Replacing {k}:{cfg.model[k]} with {config[k]}")
            cfg.model[k] = config[k]

    OmegaConf.set_struct(cfg, True)
    return cfg 


MODEL_ULRS = {
    'bert-base-uncased': 'https://cloud.tsinghua.edu.cn/f/f45fff1a308846cfa63a/?dl=1',
    'bert-large-uncased': 'https://cloud.tsinghua.edu.cn/f/6d4f38c96e8c4c16917e/?dl=1',
    'roberta-base': 'https://cloud.tsinghua.edu.cn/f/307fd141932440bc92da/?dl=1',
    'roberta-large': 'https://cloud.tsinghua.edu.cn/f/66c42c24ca304cecaf7e/?dl=1',
    'vit-base-patch16-224-in21k': 'https://cloud.tsinghua.edu.cn/f/fdf40233d9034b6a8bdc/?dl=1',
    'deit-tiny': 'https://cloud.tsinghua.edu.cn/f/b759657cb80e4bc69303/?dl=1',
    'deit-small': 'https://cloud.tsinghua.edu.cn/f/51498210e2c943dbbef1/?dl=1',
    'deit-base': 'https://cloud.tsinghua.edu.cn/f/9a26fd1aee7146e1a848/?dl=1',
    'cait-s24-224': 'https://cloud.tsinghua.edu.cn/f/bdfb12396000468b8bb9/?dl=1',
    # CLIP
    'clip': 'https://cloud.tsinghua.edu.cn/f/bd29f0537f9949e6a4fb/?dl=1', # vit-base-patch32
    'clip-vit-base-patch16': 'https://lfs.aminer.cn/misc/clip/clip-vit-base-patch16.zip',
    'clip-vit-large-patch14': 'https://lfs.aminer.cn/misc/clip/clip-vit-large-patch14.zip',
    'yolos-tiny': 'https://cloud.tsinghua.edu.cn/f/8ee048b6a1f1403d9253/?dl=1',
    'mae-vit-base': 'https://cloud.tsinghua.edu.cn/f/5ab3543f0e1d4507ad8c/?dl=1',
    'cogview-base': 'https://cloud.tsinghua.edu.cn/f/df21f6d4109b4285bfd9/?dl=1',
    'glm-large-zh': 'https://lfs.aminer.cn/misc/cogview/glm/glm-large-zh.zip',
    'glm-large-en-blank': 'https://lfs.aminer.cn/misc/cogview/glm/glm-large-en-blank.zip',
    'glm-10b-en': 'https://lfs.aminer.cn/misc/cogview/glm/glm-10b-en.zip',
    'glm-10b-zh': 'https://lfs.aminer.cn/misc/cogview/glm/glm-10b-zh.zip',
    # 'glm-large-zh': 'https://cloud.tsinghua.edu.cn/f/df21f6d4109b4285bfd9/?dl=1',
    # 'glm-large-en-blank': 'https://cloud.tsinghua.edu.cn/f/df21f6d4109b4285bfd9/?dl=1',
    'gpt-neo-1.3b': 'https://cloud.tsinghua.edu.cn/f/22e87976b5b745ad90af/?dl=1',

    # CogView2
    'coglm': 'https://lfs.aminer.cn/misc/cogview/cogview2/coglm.zip',
    'cogview2-dsr': 'https://lfs.aminer.cn/misc/cogview/cogview2/cogview2-dsr.zip',
    'cogview2-itersr': 'https://lfs.aminer.cn/misc/cogview/cogview2/cogview2-itersr.zip',
    
    # CogVideo
    'cogvideo-stage1': 'https://lfs.aminer.cn/misc/cogvideo/cogvideo-stage1.zip', 
    'cogvideo-stage2': 'https://lfs.aminer.cn/misc/cogvideo/cogvideo-stage2.zip',
    
    # DPR
    'dpr-ctx_encoder-single-nq-base': 'https://cloud.tsinghua.edu.cn/f/e5475f1211a948708baa/?dl=1',
    'dpr-question_encoder-single-nq-base': 'https://cloud.tsinghua.edu.cn/f/5c4aae7d11fc4c45a5bd/?dl=1',
    'dpr-reader-single-nq-base': 'https://cloud.tsinghua.edu.cn/f/e169889ab40d4615a34d/?dl=1',
}

def download_with_progress_bar(save_path, url):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pbar = tqdm(total=int(r.headers['Content-Length']), unit_scale=True)
            for chunk in r.iter_content(chunk_size=32 * 1024):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    pbar.update(len(chunk))

def auto_create(name, *, path=None, url=None):
    """Fetches the pre-trained model given by name, and downloads it to path."""
    if path is None:
        path = os.getenv('SAT_HOME', '~/.sat_models') # TODO Rename
    path = os.path.expanduser(path)
    file_path = os.path.join(path, name + '.zip')
    model_path = os.path.join(path, name)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    lock = FileLock(file_path + '.lock')
    with lock:
        if os.path.exists(file_path) or os.path.isdir(model_path):
            pass
        else:
            if url is None:
                url = MODEL_ULRS[name]
            print(f'Downloading models {url} into {file_path} ...')
            download_with_progress_bar(file_path, url)
        # unzip
        if not os.path.isdir(model_path):
            import zipfile
            print(f'Unzipping {file_path}...')
            f = zipfile.ZipFile(file_path, 'r')
            f.extractall(path=path) # TODO check hierarcy of folders and name consistency
            assert os.path.isdir(model_path), f'Unzip failed, or the first-level folder in zip is not {name}.'
    return model_path # must return outside the `with lock` block


class SelfAttention(torch.nn.Module):
    def __init__(self, hidden_size, num_attention_heads,
                 attention_dropout_prob, output_dropout_prob,
                 init_method, layer_id, hidden_size_per_attention_head=None, output_layer_init_method=None, bias=True,
                 hooks={}, transformer_pointer=None, params_dtype=torch.float, skip_init=False, device=torch.device('cpu')):
        super(SelfAttention, self).__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        self.hooks = hooks
        self.layer_id = layer_id
        # Per attention head and per partition values.
        world_size = get_tensor_model_parallel_world_size()
        self.hidden_size = hidden_size
        if hidden_size_per_attention_head is None:
            self.hidden_size_per_attention_head = divide(hidden_size, num_attention_heads)
        else:
            self.hidden_size_per_attention_head = hidden_size_per_attention_head
        self.num_attention_heads_per_partition = divide(num_attention_heads, world_size)
        self.inner_hidden_size = num_attention_heads * self.hidden_size_per_attention_head
        self.hidden_size_per_partition = self.hidden_size_per_attention_head * self.num_attention_heads_per_partition

        # Strided linear layer.
        # TODO: All ColumnParallelLinear layers are conformant to metaseq megatron fork
        self.query_key_value = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=3 * self.inner_hidden_size,
            bias=bias,
            gather_output=False,
            init_method=init_method,
            stride=3,
            dtype=params_dtype,
            #module=self,
            #name="query_key_value",
            #skip_init=skip_init,
            #device=device
        )
        self.attention_dropout = torch.nn.Dropout(attention_dropout_prob)

        # TODO: All RowParallelLinear layers are conformant to metaseq megatron fork
        self.dense = RowParallelLinear(
            input_size=self.inner_hidden_size,
            output_size=hidden_size,
            bias=bias,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            dtype=params_dtype,
        )
        self.output_dropout = torch.nn.Dropout(output_dropout_prob)
        
        object.__setattr__(self, 'transformer', transformer_pointer)
        assert transformer_pointer is not None

    def _transpose_for_scores(self, tensor):
        """Transpose a 3D tensor [b, s, np*hn] into a 4D tensor with
        size [b, np, s, hn].
        """
        new_tensor_shape = tensor.size()[:-1] + \
                           (self.num_attention_heads_per_partition,
                            self.hidden_size_per_attention_head)
        tensor = tensor.view(*new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)

    def forward(self, hidden_states, mask, *args, **kw_args):
        if 'attention_forward' in self.hooks:
            return self.hooks['attention_forward'](hidden_states, mask, **kw_args)
        else:
            return HOOKS_DEFAULT['attention_forward'](self, hidden_states, mask, **kw_args)


class CrossAttention(torch.nn.Module):
    """Parallel cross-attention layer for Transformer"""

    def __init__(self, hidden_size, num_attention_heads, attention_dropout_prob, output_dropout_prob, init_method,
                 layer_id, hidden_size_per_attention_head=None, output_layer_init_method=None, bias=True, hooks={},
                 transformer_pointer=None, params_dtype=torch.float, skip_init=False, device=torch.device('cpu')):
        super().__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        self.hooks = hooks
        self.layer_id = layer_id
        # Per attention head and per partition values.
        world_size = get_tensor_model_parallel_world_size()
        self.hidden_size = hidden_size
        if hidden_size_per_attention_head is None:
            self.hidden_size_per_attention_head = divide(hidden_size, num_attention_heads)
        else:
            self.hidden_size_per_attention_head = hidden_size_per_attention_head
        self.num_attention_heads_per_partition = divide(num_attention_heads, world_size)
        self.inner_hidden_size = num_attention_heads * self.hidden_size_per_attention_head
        self.hidden_size_per_partition = self.hidden_size_per_attention_head * self.num_attention_heads_per_partition
        # Strided linear layer.
        self.query = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=self.inner_hidden_size,
            bias=bias,
            gather_output=False,
            init_method=init_method,
            dtype=params_dtype
        )
        self.key_value = ColumnParallelLinear(
            input_size=hidden_size,
            output_size=2 * self.inner_hidden_size,
            bias=bias,
            gather_output=False,
            init_method=init_method,
            stride=2,
            dtype=params_dtype)
        # Dropout. Note that for a single iteration, this layer will generate
        # different outputs on different number of parallel partitions but
        # on average it should not be partition dependent.
        self.attention_dropout = torch.nn.Dropout(attention_dropout_prob)

        # Output.
        self.dense = RowParallelLinear(
            input_size=self.inner_hidden_size,
            output_size=hidden_size,
            bias=bias,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            dtype=params_dtype
        )
        self.output_dropout = torch.nn.Dropout(output_dropout_prob)

        object.__setattr__(self, 'transformer', transformer_pointer)
        assert transformer_pointer is not None

    def _transpose_for_scores(self, tensor):
        """Transpose a 3D tensor [b, s, np*hn] into a 4D tensor with
        size [b, np, s, hn].
        """
        new_tensor_shape = tensor.size()[:-1] + \
                           (self.num_attention_heads_per_partition,
                            self.hidden_size_per_attention_head)
        tensor = tensor.view(*new_tensor_shape)
        return tensor.permute(0, 2, 1, 3)

    def forward(self, hidden_states, cross_attention_mask, encoder_outputs, **kw_args):
        # hidden_states: [b, s, h]
        if 'cross_attention_forward' in self.hooks:
            return self.hooks['cross_attention_forward'](hidden_states, cross_attention_mask, encoder_outputs, **kw_args)
        else:
            return HOOKS_DEFAULT['cross_attention_forward'](self, hidden_states, cross_attention_mask, encoder_outputs, **kw_args)


class MLP(torch.nn.Module):
    def __init__(self, hidden_size, output_dropout_prob, init_method, inner_hidden_size=None,
                 output_layer_init_method=None, layer_id=None, hooks={}, bias=True, activation_func=gelu, transformer_pointer=None, params_dtype=torch.float, skip_init=False, device=torch.device('cpu')):
        super(MLP, self).__init__()
        self.layer_id = layer_id
        self.activation_func = activation_func
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        self.hooks = hooks
        # Project to 4h.
        self.hidden_size = hidden_size
        if inner_hidden_size is None:
            inner_hidden_size = 4 * hidden_size
        self.inner_hidden_size = inner_hidden_size
        self.dense_h_to_4h = ColumnParallelLinear(
            input_size=self.hidden_size,
            output_size=self.inner_hidden_size,
            bias=bias,
            gather_output=False,
            init_method=init_method,
            dtype=params_dtype,
        )
        # Project back to h.
        self.dense_4h_to_h = RowParallelLinear(
            input_size=self.inner_hidden_size,
            output_size=self.hidden_size,
            bias=bias,
            input_is_parallel=True,
            init_method=output_layer_init_method,
            dtype=params_dtype,
        )
        self.dropout = torch.nn.Dropout(output_dropout_prob)
        object.__setattr__(self, 'transformer', transformer_pointer)
        assert transformer_pointer is not None
        

    def forward(self, hidden_states, **kw_args):
        if 'mlp_forward' in self.hooks:
            output = self.hooks['mlp_forward'](hidden_states, **kw_args)
        else:
            output = HOOKS_DEFAULT['mlp_forward'](self, hidden_states, **kw_args)

        if self.training:
            output = self.dropout(output)
        return output


class BaseTransformerLayer(torch.nn.Module):
    def __init__(
            self,
            hidden_size,
            num_attention_heads,
            attention_dropout_prob,
            output_dropout_prob,
            layernorm_epsilon,
            init_method,
            layer_id,
            inner_hidden_size=None,
            hidden_size_per_attention_head=None,
            output_layer_init_method=None,
            layernorm_order='pre',
            layernorm=LayerNorm,
            is_decoder=False,
            use_bias=True,
            activation_func=gelu,
            hooks={},
            transformer_pointer=None,
            params_dtype=torch.float,
            skip_init=False,
            device=torch.device('cpu')
    ):
        super(BaseTransformerLayer, self).__init__()
        # Set output layer initialization if not provided.
        if output_layer_init_method is None:
            output_layer_init_method = init_method
        self.layer_id = layer_id
        self.is_decoder = is_decoder
        self.layernorm_order = layernorm_order
        self.hooks = hooks
        object.__setattr__(self, 'transformer', transformer_pointer)
        assert transformer_pointer is not None

        # Layernorm on the input data.
        self.input_layernorm = layernorm(hidden_size, eps=layernorm_epsilon)

        # Self attention.
        self.attention = SelfAttention(
            hidden_size,
            num_attention_heads,
            attention_dropout_prob,
            output_dropout_prob,
            init_method,
            layer_id,
            hidden_size_per_attention_head=hidden_size_per_attention_head,
            output_layer_init_method=output_layer_init_method,
            bias=use_bias,
            hooks=hooks,
            transformer_pointer=transformer_pointer,
            params_dtype=params_dtype,
            skip_init=skip_init,
            device=device
        )

        # Layernorm on the input data.
        self.post_attention_layernorm = layernorm(hidden_size, eps=layernorm_epsilon)
        if self.layernorm_order == 'sandwich':
            self.third_layernorm = layernorm(hidden_size, eps=layernorm_epsilon)
            self.fourth_layernorm = layernorm(hidden_size, eps=layernorm_epsilon)

        # Cross attention.
        if self.is_decoder:
            self.cross_attention = CrossAttention(
                hidden_size,
                num_attention_heads,
                attention_dropout_prob,
                output_dropout_prob,
                init_method,
                layer_id,
                hidden_size_per_attention_head=hidden_size_per_attention_head,
                output_layer_init_method=output_layer_init_method,
                bias=use_bias,
                hooks=hooks,
                transformer_pointer=transformer_pointer,
                params_dtype=params_dtype
            )
            self.post_cross_attention_layernorm = layernorm(hidden_size, eps=layernorm_epsilon)

        # MLP
        self.mlp = MLP(
            hidden_size,
            output_dropout_prob,
            init_method,
            inner_hidden_size=inner_hidden_size,
            output_layer_init_method=output_layer_init_method,
            bias=use_bias,
            layer_id=layer_id,
            activation_func=activation_func,
            hooks=hooks,
            transformer_pointer=transformer_pointer,
            params_dtype=params_dtype,
            skip_init=skip_init,
            device=device
        )

    def forward(self, hidden_states, mask, *args, **kw_args):
        return HOOKS_DEFAULT['layer_forward'](self, hidden_states, mask, *args, **kw_args)


class BaseTransformer(torch.nn.Module):
    def __init__(self,
                 num_layers,
                 vocab_size,
                 hidden_size,
                 num_attention_heads,
                 max_sequence_length,
                 embedding_dropout_prob,
                 attention_dropout_prob,
                 output_dropout_prob,
                 checkpoint_activations,
                 checkpoint_num_layers=1,
                 layernorm_epsilon=1.0e-5,
                 init_method_std=0.02,
                 inner_hidden_size=None,
                 hidden_size_per_attention_head=None,
                 layernorm_order='pre',
                 parallel_output=True,
                 is_decoder=False,
                 use_bias=True,
                 activation_func=gelu,
                 layernorm=LayerNorm,
                 init_method=None,
                 use_final_layernorm=True,
                 hooks={},
                 params_dtype=torch.float,
                 skip_init=False,
                 device=torch.device('cpu')
                 ):
        super(BaseTransformer, self).__init__()

        # recording parameters
        self.is_decoder = is_decoder
        self.parallel_output = parallel_output
        self.checkpoint_activations = checkpoint_activations
        self.checkpoint_num_layers = checkpoint_num_layers
        self.max_sequence_length = max_sequence_length
        self.layernorm_order = layernorm_order
        self.hooks = copy.copy(hooks)  # hooks will be updated each forward
        object.__setattr__(self, 'transformer', self) # to give the default hooks the same api as outer hooks

        # create embedding parameters
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)

        #self.word_embeddings = VocabParallelEmbedding(
        #    num_embeddings=vocab_size, embedding_dim=hidden_size, 
        #    params_dtype=params_dtype, skip_init=skip_init, device=device)
        # TODO: skip_init and device shouldn't be needed
        self.word_embeddings = VocabParallelEmbedding(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            padding_idx=None,   # Not used in GLM
            dtype=params_dtype,
        )

        self.position_embeddings = torch.nn.Embedding(max_sequence_length, hidden_size)
        torch.nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=init_method_std)

        # create all layers
        if init_method is None:
            self.output_layer_init_method = scaled_init_method(init_method_std, num_layers)
            self.init_method = unscaled_init_method(init_method_std)
        else:
            self.output_layer_init_method = init_method
            self.init_method = init_method

        def get_layer(layer_id):
            return BaseTransformerLayer(
                hidden_size,
                num_attention_heads,
                attention_dropout_prob,
                output_dropout_prob,
                layernorm_epsilon,
                self.init_method,
                layer_id,
                inner_hidden_size=inner_hidden_size,
                hidden_size_per_attention_head=hidden_size_per_attention_head,
                output_layer_init_method=self.output_layer_init_method,
                is_decoder=self.is_decoder,
                layernorm_order=layernorm_order,
                layernorm=layernorm,
                use_bias=use_bias,
                activation_func=activation_func,
                hooks=self.hooks,
                transformer_pointer=self,
                params_dtype=params_dtype,
                skip_init=skip_init,
                device=device
            )

        self.layers = torch.nn.ModuleList(
            [get_layer(layer_id) for layer_id in range(num_layers)])

        # Final layer norm before output.
        self.use_final_layernorm = use_final_layernorm
        if use_final_layernorm:
            self.final_layernorm = layernorm(hidden_size, eps=layernorm_epsilon)

    def forward(self, input_ids, position_ids, attention_mask, *,
                output_hidden_states=False, **kw_args):
        # sanity check
        assert len(input_ids.shape) >= 2
        batch_size, query_length = input_ids.shape[:2]

        if attention_mask is None:
            attention_mask = torch.ones(1, 1, device=input_ids.device).type_as(
                next(self.parameters())
            )  # None means full attention
        assert len(attention_mask.shape) == 2 or \
               len(attention_mask.shape) == 4 and attention_mask.shape[1] == 1

        # initial output_cross_layer might be generated by word/position_embedding_forward
        output_cross_layer = {}

        # embedding part
        if 'word_embedding_forward' in self.hooks:
            hidden_states = self.hooks['word_embedding_forward'](input_ids, output_cross_layer=output_cross_layer, **kw_args)
        else:  # default
            hidden_states = HOOKS_DEFAULT['word_embedding_forward'](self, input_ids, output_cross_layer=output_cross_layer,**kw_args)

        if 'position_embedding_forward' in self.hooks:
            position_embeddings = self.hooks['position_embedding_forward'](position_ids, output_cross_layer=output_cross_layer, **kw_args)
        else:
            assert len(position_ids.shape) <= 2
            assert position_ids.shape[-1] == hidden_states.shape[1], (position_ids.shape, hidden_states.shape)
            position_embeddings = HOOKS_DEFAULT['position_embedding_forward'](self, position_ids, output_cross_layer=output_cross_layer, **kw_args)
        if position_embeddings is not None:
            hidden_states = hidden_states + position_embeddings
        hidden_states = self.embedding_dropout(hidden_states)

        output_per_layers = []
        if self.checkpoint_activations:
            # define custom_forward for checkpointing
            def custom(start, end, kw_args_index, cross_layer_index):
                def custom_forward(*inputs):
                    layers_ = self.layers[start:end]
                    x_, mask = inputs[0], inputs[1]

                    # recover kw_args and output_cross_layer
                    flat_inputs = inputs[2:]
                    kw_args, output_cross_layer = {}, {}
                    for k, idx in kw_args_index.items():
                        kw_args[k] = flat_inputs[idx]
                    for k, idx in cross_layer_index.items():
                        output_cross_layer[k] = flat_inputs[idx]
                    # -----------------

                    output_per_layers_part = []
                    for i, layer in enumerate(layers_):
                        output_this_layer_obj, output_cross_layer_obj = {}, {}
                        if 'layer_forward' in self.hooks:
                            layer_ret = self.hooks['layer_forward'](
                                x_, mask, layer_id=layer.layer_id,
                                **kw_args, **output_cross_layer,
                                output_this_layer=output_this_layer_obj,
                                output_cross_layer=output_cross_layer_obj
                            )
                        else:
                            layer_ret = layer(
                                x_, mask, layer_id=layer.layer_id,
                                **kw_args, **output_cross_layer,
                                output_this_layer=output_this_layer_obj,
                                output_cross_layer=output_cross_layer_obj
                            )
                        if isinstance(layer_ret, tuple):
                            layer_ret = layer_ret[0] # for legacy API
                        x_, output_this_layer, output_cross_layer = layer_ret, output_this_layer_obj, output_cross_layer_obj
                        if output_hidden_states:
                            output_this_layer['hidden_states'] = x_
                        output_per_layers_part.append(output_this_layer)

                    # flatten for re-aggregate keywords outputs
                    flat_outputs = []
                    for output_this_layer in output_per_layers_part:
                        for k in output_this_layer:
                            # TODO add warning for depth>=2 grad tensors
                            flat_outputs.append(output_this_layer[k])
                            output_this_layer[k] = len(flat_outputs) - 1
                    for k in output_cross_layer:
                        flat_outputs.append(output_cross_layer[k])
                        output_cross_layer[k] = len(flat_outputs) - 1
                    # --------------------

                    return (x_, output_per_layers_part, output_cross_layer, *flat_outputs)
                return custom_forward

            # prevent to lose requires_grad in checkpointing.
            # To save memory when only finetuning the final layers, don't use checkpointing.
            if self.training:
                hidden_states.requires_grad_(True)

            l, num_layers = 0, len(self.layers)
            chunk_length = self.checkpoint_num_layers
            output_this_layer = []
            while l < num_layers:
                args = [hidden_states, attention_mask]
                # flatten kw_args and output_cross_layer
                flat_inputs, kw_args_index, cross_layer_index = [], {}, {}
                for k, v in kw_args.items():
                    flat_inputs.append(v)
                    kw_args_index[k] = len(flat_inputs) - 1
                for k, v in output_cross_layer.items():
                    flat_inputs.append(v)
                    cross_layer_index[k] = len(flat_inputs) - 1
                # --------------------
                # TODO: Get rid of all deepspeed calls
                #hidden_states, output_per_layers_part, output_cross_layer, *flat_outputs = \
                #    checkpoint(custom(l, l + chunk_length, kw_args_index, cross_layer_index), *args, *flat_inputs)
                hidden_states, output_per_layers_part, output_cross_layer, *flat_outputs = custom(l, l + chunk_length, kw_args_index, cross_layer_index)
                
                # recover output_per_layers_part, output_cross_layer
                for output_this_layer in output_per_layers_part:
                    for k in output_this_layer:
                        output_this_layer[k] = flat_outputs[output_this_layer[k]]
                for k in output_cross_layer:
                    output_cross_layer[k] = flat_outputs[output_cross_layer[k]]
                # --------------------

                output_per_layers.extend(output_per_layers_part)
                l += chunk_length
        else:
            output_this_layer = []
            for i, layer in enumerate(self.layers):
                args = [hidden_states, attention_mask]

                output_this_layer_obj, output_cross_layer_obj = {}, {}

                if 'layer_forward' in self.hooks: # customized layer_forward
                    layer_ret = self.hooks['layer_forward'](*args,
                        layer_id=torch.tensor(i),
                        **kw_args,
                        position_ids=position_ids,
                        **output_cross_layer,
                        output_this_layer=output_this_layer_obj, output_cross_layer=output_cross_layer_obj
                    )
                else:
                    layer_ret = layer(*args, layer_id=torch.tensor(i), **kw_args, **output_cross_layer,
                        output_this_layer=output_this_layer_obj, output_cross_layer=output_cross_layer_obj)
                if isinstance(layer_ret, tuple):
                    layer_ret = layer_ret[0] # for legacy API
                hidden_states, output_this_layer, output_cross_layer = layer_ret, output_this_layer_obj, output_cross_layer_obj

                if output_hidden_states:
                    output_this_layer['hidden_states'] = hidden_states
                output_per_layers.append(output_this_layer)

        # Final layer norm.
        if self.use_final_layernorm:
            logits = self.final_layernorm(hidden_states)
        else:
            logits = hidden_states

        logits = copy_to_model_parallel_region(logits)
        if 'final_forward' in self.hooks:
            logits_parallel = self.hooks['final_forward'](logits, **kw_args)
        else:
            logits_parallel = HOOKS_DEFAULT['final_forward'](self, logits, **kw_args)

        if not self.parallel_output:
            logits_parallel = gather_from_model_parallel_region(logits_parallel)

        outputs = [logits_parallel]
        outputs.extend(output_per_layers)
        
        return outputs


class BaseModel(torch.nn.Module):
    def __init__(self, cfg, transformer=None, params_dtype=torch.float, **kwargs):
        super(BaseModel, self).__init__()
        self.mixins = torch.nn.ModuleDict()
        self.collect_hooks_()
        if transformer is not None:
            self.transformer = transformer
        else:
            self.transformer = BaseTransformer(
                num_layers=cfg.model.num_layers,
                vocab_size=cfg.model.vocab_size,
                hidden_size=cfg.model.hidden_size,
                num_attention_heads=cfg.model.num_attention_heads,
                max_sequence_length=cfg.model.max_sequence_length,
                embedding_dropout_prob=cfg.model.hidden_dropout,
                attention_dropout_prob=cfg.model.attention_dropout,
                output_dropout_prob=cfg.model.hidden_dropout,
                inner_hidden_size=cfg.model.inner_hidden_size,
                hidden_size_per_attention_head=cfg.model.hidden_size_per_attention_head,
                #checkpoint_activations=cfg.model.checkpoint_activations, # TODO: Deepspeed turned off
                checkpoint_activations=False,
                #checkpoint_num_layers=cfg.model.checkpoint_num_layers, #TODO: Deepspeed turned off
                checkpoint_num_layers=1,
                layernorm_order=cfg.model.layernorm_order,
                hooks=self.hooks,
                params_dtype=params_dtype,
                skip_init=cfg.model.skip_init,
                device=torch.cuda.current_device() if cfg.model.use_gpu_initialization else torch.device('cpu'),
                **kwargs
            )

    def reinit(self, mixin_names=None):  # will be called when loading model, None means all
        # if some mixins are loaded, overrides this function
        for k, m in self.mixins.items():
            if k in mixin_names or mixin_names is None:
                m.reinit(self)

    def add_mixin(self, name, new_mixin, reinit=False):
        assert name not in self.mixins
        assert isinstance(new_mixin, BaseMixin)

        self.mixins[name] = new_mixin  # will auto-register parameters
        object.__setattr__(new_mixin, 'transformer', self.transformer)  # cannot use pytorch set_attr

        self.collect_hooks_()
        if reinit:
            new_mixin.reinit(self)  # also pass current mixins

    def del_mixin(self, name):
        assert name in self.mixins
        del self.mixins[name]
        self.collect_hooks_()

    def get_mixin(self, name):
        return self.mixins[name]

    def forward(self, *args, **kwargs):
        # update hooks as the current model (overrided forwards)
        # Attention! the transformer might be shared by multiple models
        self.transformer.hooks.clear()
        self.transformer.hooks.update(self.hooks)
        return self.transformer(*args, **kwargs)

    def collect_hooks_(self):
        names = list(HOOKS_DEFAULT.keys())
        hooks = {}
        hook_origins = {}
        for name in names:
            if hasattr(self, name):
                hooks[name] = getattr(self, name)
                hook_origins[name] = 'model'

            for mixin_name, m in self.mixins.items():
                if hasattr(m, name):
                    if hasattr(getattr(m, name), 'non_conflict'):
                        if name in hooks:
                            old_impl = hooks[name]
                        elif name == 'attention_fn': # the only hook without self
                            old_impl = HOOKS_DEFAULT[name]
                        else:
                            old_impl = partial(HOOKS_DEFAULT[name], self)
                        old_origin = hook_origins.get(name, 'default')
                        hooks[name] = partial(getattr(m, name), old_impl=old_impl)
                        hook_origins[name] = mixin_name + ' -> ' + old_origin
                    elif name in hooks: # if this hook name is already registered
                        raise ValueError(f'Hook {name} conflicts at {mixin_name} and {hook_origins[name]}.')
                    else: # new hook
                        hooks[name] = getattr(m, name)
                        hook_origins[name] = mixin_name

        self.hooks = hooks
        self.hook_origins = hook_origins
        return hooks

    def disable_untrainable_params(self):
        pass

    @classmethod
    def from_pretrained(cls, cfg, name, *, home_path=None, url=None, prefix=''):
        model_path = auto_create(name, path=home_path, url=url)
        cfg = update_args_with_file(cfg, path=os.path.join(model_path, 'model_config.json'))
        model = get_model(cfg, cls)
        load_checkpoint(model, cfg, load_path=model_path, prefix=prefix)
        return model, args


class CachedAutoregressiveMixin(BaseMixin):
    def __init__(self):
        super().__init__()     
           
    @non_conflict
    def attention_fn(self, q, k, v, mask, dropout_fn, mems=None, cross_attention=False, old_impl=standard_attention,
                     **kw_args):
        if not cross_attention:
            mem = mems[kw_args['layer_id']] if mems is not None else None # 2, batch, head, seqlen, hidden_size
            b, nh, seq_len, hidden_size = k.shape

            cache_kv = torch.stack((k, v)).permute(1, 3, 0, 2, 4).detach().contiguous().view(b, seq_len, nh * hidden_size * 2)
            kw_args['output_this_layer']['mem_kv'] = cache_kv

            if mem is not None: # the first time, mem is None
                # might change batch_size
                mem = mem.expand(b, -1, -1).reshape(b, mem.shape[1], 2, nh, hidden_size).permute(2, 0, 3, 1, 4)
                memk, memv = mem[0], mem[1]
                k = torch.cat((memk, k), dim=2)
                v = torch.cat((memv, v), dim=2)
        return old_impl(q, k, v, mask, dropout_fn, cross_attention=cross_attention, mems=mems, **kw_args)


class CachedAutoregressiveModel(BaseModel):
    def __init__(self, args, transformer=None):
        super().__init__(args, transformer=transformer)
        self.add_mixin('auto-regressive', CachedAutoregressiveMixin())


class BlockPositionEmbeddingMixin(BaseMixin):
    def __init__(self, max_sequence_length, hidden_size, init_method_std=0.02):
        super(BlockPositionEmbeddingMixin, self).__init__()
        self.max_sequence_length = max_sequence_length
        self.hidden_size = hidden_size
        self.block_position_embeddings = torch.nn.Embedding(max_sequence_length, hidden_size)
        torch.nn.init.normal_(self.block_position_embeddings.weight, mean=0.0, std=init_method_std)
    
    def position_embedding_forward(self, position_ids, **kwargs):
        position_ids, block_position_ids = position_ids[:, 0], position_ids[:, 1]
        position_embeddings = self.transformer.position_embeddings(position_ids)
        block_position_embeddings = self.block_position_embeddings(block_position_ids)
        return position_embeddings + block_position_embeddings


@dataclass
class GLMModelConfig(MetaseqDataclass):
    num_layers: int = field(
        default=24,
        metadata={"help": "Number of decoder layers"}
    )
    hidden_size: int = field(
        default=1024,
        metadata={"help": "Transformer hidden dim size"}
    )
    num_attention_heads: int = field(
        default=16,
        metadata={"help": "Number of transformer attention heads"}
    )
    vocab_size: int = field(
        default=0,
        metadata={"help": "Vocab size for tokenization"}
    )
    max_sequence_length: int = field(
        default=512,
        metadata={"help": "Max number of position embeddings to use"}
    )
    layernorm_order: str = field(
        default="pre",
        metadata={"help": "Order of layernorm (post, pre, sandwich)"}
    )
    inner_hidden_size: Optional[int] = field(
        default=None,
        metadata={"help": "Inner hidden size in MLP, None meaning 4 * hidden size"}
    )
    hidden_size_per_attention_head: Optional[int] = field(
        default=None,
        metadata={"help": "Hidden size per attention head in self and cross attention. None means hidden_sized / num_attention_heads"}
    )
    skip_init: bool = field(
        default=False,
        metadata={"help": "Skip model initialization"}
    )
    use_gpu_initialization: bool = field(
        default=False,
        metadata={"help": "Initialize model on GPU"}
    )
    layernorm_epsilon: float = field(
        default=1e-5,
        metadata={"help": "Layer norm epsilon"}
    )
    hidden_dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout prob for hidden state"}
    )
    attention_dropout: float = field(
        default=0.1,
        metadata={"help": "Dropout prob for attention weights"}
    )
    make_vocab_size_divisible_by: int = field(
        default=128,
        metadata={"help": "Pad the vocab size to be divisible by this value"}
    )
    sandwich_ln: bool = field(
        default=False,
        metadata={"help": "Add sandwich ln in cogview"}
    )


    block_lm: bool = field(
        default=False,
        metadata={"help": "Whether to use the BlockLM pre-training"}
    )
    masked_lm: bool = field(
        default=False,
        metadata={"help": "Whether to use the MLM objective"}
    )
    bert_prob: float = field(
        default=0.5,
        metadata={"help": ""}
    )
    gpt_infill_prob: float = field(
        default=0.5,
        metadata={"help": ""}
    )
    gpt_min_ratio: float = field(
        default=0.5,
        metadata={"help": ""}
    )
    gap_sentence_prob: float = field(
        default=0.0,
        metadata={"help": ""}
    )
    gap_sentence_ratio: float = field(
        default=0.15,
        metadata={"help": ""}
    )
    avg_block_length: int = field(
        default=3,
        metadata={"help": ""}
    )
    short_seq_prob: float = field(
        default=0.0,
        metadata={"help": ""}
    )
    single_span_prob: float = field(
        default=0.0,
        metadata={"help": ""}
    )
    task_mask: bool = field(
        default=False,
        metadata={"help": "Use different mask for generation and blank infilling"}
    )
    no_shuffle_block: bool = field(
        default=False,
        metadata={"help": "Not shuffle the blocks when filling the blank"}
    )
    no_block_position: bool = field(
        default=False,
        metadata={"help": "Use (rought) absolute positions instead of block positions"}
    )
    sentinel_token: bool = field(
        default=False,
        metadata={"help": "Use sentinel (mask) tokens to replace 2d position encoding"}
    )
    block_mask_prob: float = field(
        default=0.0,
        metadata={"help": ""}
    )
    context_mask_ratio: float = field(
        default=0.0,
        metadata={"help": ""}
    )
    random_position: bool = field(
        default=False,
        metadata={"help": "Use random start position to cover all the position embeddings"}
    )
    cloze_eval: bool = field(
        default=False,
        metadata={"help": "Evaluation dataset with cloze task"}
    )
    old_checkpoint: bool = field(
        default=False,
        metadata={"help": "Loading the checkpoint from old library"}
    )
    tokenizer_model_type: Optional[str] = field(
        default=None,
        metadata={"help": "Model type to use for sentencepiece tokenization [bpe, char, unigram, word]"}
    )



@register_model("glm_large", dataclass=GLMModelConfig)
class GLMModel(BaseModel):
    def __init__(self, cfg, transformer=None, parallel_output=True):
        super().__init__(cfg, transformer=transformer, parallel_output=parallel_output)
        self.add_mixin('block_position_embedding', 
            BlockPositionEmbeddingMixin(cfg.model.max_sequence_length, cfg.model.hidden_size)
        )

    @classmethod
    def add_args(cls, parser):
        """Add model-specific arguments to the parser."""                                                                       
        dc = getattr(cls, "__dataclass", None) 
        if dc is not None:                                                                                                              
            # do not set defaults so that settings defaults from various architectures still works                                      
            gen_parser_from_dataclass(parser, dc(), delete_default=True)
