from dataclasses import _MISSING_TYPE, dataclass, field
from typing import Any, Optional, List

import torch
from omegaconf import II, MISSING

from metaseq.dataclass.configs import MetaseqDataclass
from metaseq.dataclass.constants import (
    DATASET_IMPL_CHOICES,
    DDP_BACKEND_CHOICES,
    LOG_FORMAT_CHOICES,
    ZERO_SHARDING_CHOICES,
    CLIP_GRAD_NORM_TYPE_CHOICES,
)

# NOTE: This file is for config reference only. Do not use

@dataclass
class SwissModelConfig(MetaseqDataclass):
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
    inner_hidden_size: int = field(
        default=None,
        metadata={"help": "Inner hidden size in MLP, None meaning 4 * hidden size"}
    )
    hidden_size_per_attention_head: int = field(
        default=None,
        metadata={"help": "Hidden size per attention head in self and cross attention. None means hidden_sized / num_attention_heads"}
    )
    model_parallel_size: int = field(
        default=1,
        metadata={"help": "Size of the model parallel"}
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


@dataclass
class SwissDistributedTrainingConfig(MetaseqDataclass):
    distributed_world_size: int = field(
        default=max(1, torch.cuda.device_count()),
        metadata={
            "help": "total number of GPUs across all nodes (default: all visible GPUs)"
        },
    )
    distributed_rank: Optional[int] = field(
        default=0, metadata={"help": "rank of the current worker"}
    )
    distributed_backend: str = field(
        default="nccl", metadata={"help": "distributed backend"}
    )
    distributed_init_method: Optional[str] = field(
        default=None,
        metadata={
            "help": "typically tcp://hostname:port that will be used to "
            "establish initial connetion"
        },
    )
    distributed_port: int = field(
        default=-1,
        metadata={
            "help": "port number (not required if using --distributed-init-method)"
        },
    )
    device_id: int = field(
        default=0,
        metadata={
            "help": "which GPU to use (usually configured automatically)",
            "argparse_alias": "--local_rank",
        },
    )
    distributed_no_spawn: bool = field(
        default=False,
        metadata={
            "help": "do not spawn multiple processes even if multiple GPUs are visible"
        },
    )
    ddp_backend: DDP_BACKEND_CHOICES = field(
        default="pytorch_ddp", metadata={"help": "DistributedDataParallel backend"}
    )
    bucket_cap_mb: int = field(
        default=25, metadata={"help": "bucket size for reduction"}
    )
    fix_batches_to_gpus: bool = field(
        default=False,
        metadata={
            "help": "don't shuffle batches between GPUs; this reduces overall "
            "randomness and may affect precision but avoids the cost of re-reading the data"
        },
    )
    find_unused_parameters: bool = field(
        default=False,
        metadata={
            "help": "disable unused parameter detection (not applicable to "
            "--ddp-backend=legacy_ddp)"
        },
    )
    fast_stat_sync: bool = field(
        default=False,
        metadata={"help": "[deprecated] this is now defined per Criterion"},
    )
    heartbeat_timeout: int = field(
        default=-1,
        metadata={
            "help": "kill the job if no progress is made in N seconds; "
            "set to -1 to disable"
        },
    )
    broadcast_buffers: bool = field(
        default=False,
        metadata={
            "help": "Copy non-trainable parameters between GPUs, such as "
            "batchnorm population statistics"
        },
    )
    zero_sharding: ZERO_SHARDING_CHOICES = field(
        default="none", metadata={"help": "ZeRO sharding"}
    )
    fp16: bool = field(
        default=False,
        metadata={"help": "Use fp16"}
    )
    bf16: bool = field(
        default=False,
        metadata={"help": "Use bf16"}
    )
    # configuration for --ddp-backend=fully_sharded
    no_reshard_after_forward: bool = field(
        default=False,
        metadata={"help": "don't reshard parameters after forward pass"},
    )
    fp32_reduce_scatter: bool = field(
        default=False,
        metadata={"help": "reduce-scatter grads in FP32"},
    )
    cpu_offload: bool = field(
        default=False, metadata={"help": "offload FP32 params to CPU"}
    )
    use_sharded_state: Optional[bool] = field(
        default=False, metadata={"help": "load and save local state dict"}
    )
    gradient_predivide_factor: Optional[float] = field(
        default=None,
        metadata={"help": "factor to predivide gradients before reducee scatter"},
    )


@dataclass
class SwissTrainingConfig(MetaseqDataclass):
    experiment_name: str = field(
        default="DefaultModel",
        metadata={"help": "Experiment name for summary and checkpointing"}
    )
    train_iters: int = field(
        default=1000000,
        metadata={"help": "Total number of iterations to train over all training runs"}
    )
    batch_size: int = field(
        default=4,
        metadata={"help": "Batch size on a single GPU"}
    )
    #lr: float = field(
    #    default=1.0e-4,
    #    metadata={"help": "Initial learning rate"}
    #)
    mode: str = field(
        default="pretrain",
        metadata={"help": "pretrain, finetune or inference"}
    )
    seed: int = field(
        default=42069,
        metadata={"help": "Random seed"}
    )
    zero_stage: int = II("distributed_training.zero_sharding")     
    checkpoint_activations: bool = field(
        default=False,
        metadata={"help": "Checkpoint activations during training"}
    )
    checkpoint_num_layers: int = field(
        default=1,
        metadata={"help": "Chunk size (number of layers) for checkpointing"}
    )
    fp16: bool = II("distributed_training.fp16")
    bf16: bool = II("distributed_training.bf16")
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Run optimizer after every gradient_accumulation_steps backwards"}
    )
    epochs: int = field(
        default=None,
        metadata={"help": "Number of training epochs"}
    )
    log_interval: int = field(
        default=50,
        metadata={"help": "Report interval"}
    )
    summary_dir: str = field(
        default="",
        metadata={"help": "The directory to store the summary"}
    )
    save_args: bool = field(
        default=False,
        metadata={"help": "Save args corresponding to the experiment_name"}
    )
    lr_decay_iters: int = field(
        default=None,
        metadata={"help": "If None defaults to train_iters * epochs"}
    )
    lr_decay_style: str = field(
        default="linear",
        metadata={"help": "Choose from constant, linear, cosine, exponential"}
    )
    lr_decay_ratio: float = field(
        default=0.1,
        metadata={"help": "Learning rate decay ratio"}
    )
    warmup: float = field(
        default=0.01,
        metadata={"help": "Percentage of data to warmup on"}
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "Weight decay coefficient for L2 regularization"}
    )
    save: str = field(
        default=None,
        metadata={"help": "Output directory to save checkpoints to"}
    )
    load: str = field(
        default=None,
        metadata={"help": "Path to a directory containing a model checkpoint"}
    )
    save_interval: int = field(
        default=5000,
        metadata={"help": "Number of iterations between saves"}
    )
    no_save_rng: bool = field(
        default=False,
        metadata={"help": "Do not save the current rng state"}
    )
    no_load_rng: bool = field(
        default=False,
        metadata={"help": "Do not load rng state when loading checkpoint"}
    )
    resume_dataloader: bool = field(
        default=False,
        metadata={"help": "Resume the dataloader when resuming training"}
    )
    distributed_backend: str = field(
        default="nccl",
        metadata={"help": "Which backend to use for distributed training. One of gloo or nccl"}
    )
    local_rank: int = field(
        default=None,
        metadata={"help": "Local rank pased from distributed launcher"}
    )
    exit_interval: int = field(
        default=None,
        metadata={"help": "Exit the program after this many new iterations"}
    )


@dataclass
class SwissEvaluationConfig(MetaseqDataclass):
    eval_batch_size: int = field(
        default=None,
        metadata={"help": "Data loader batch size for evaluation dataset"}
    )
    eval_iters: int = field(
        default=100,
        metadata={"help": "Number of iterations to run for evaluation"}
    )
    eval_interval: int = field(
        default=None,
        metadata={"help": "Interval between running evaluation on validation set"}
    )
    strict_eval: bool = field(
        default=False,
        metadata={"help": "Won't enlarge or randomly map eval ata, and eval full eval data"}
    )


@dataclass
class SwissDataConfig(MetaseqDataclass):
    # TODO: Several of these fields are lists by default in GLM
    train_data: str = field(
        default=None,
        metadata={"help": "Whitespace separated filenames or corpora names for training"}
    )
    train_data_weights: str = field(
        default=None,
        metadata={"help": "Scaling factors for different train data, must be the same number"}
    )
    iterable_dataset: bool = field(
        default=False,
        metadata={"help": "Iterable"}
    )
    valid_data: str = field(
        default=None,
        metadata={"help": "Filename for validation data"}
    )
    test_data: str = field(
        default=None,
        metadata={"help": "Filename for test data"}
    )
    split: str = field(
        default=None,
        metadata={"help": "Comma separated list of proportions for training"}
    )
    num_workers: int = field(
        default=1,
        metadata={"help": " Number of workers to use for dataloading"}
    )
    block_size: int = field(
        default=10000,
        metadata={"help": "Size of lock to reduce memory in dataset, ignore it for most users"}
    )


@dataclass
class SwissTextGenerationConfig(MetaseqDataclass):
    temperature: float = field(
        default=1.0,
        metadata={"help": "Sampling temperature"}
    )
    top_p: float = field(
        default=0.0,
        metadata={"help": "Top p for sampling"}
    )
    top_k: int = field(
        default=0,
        metadata={"help": "Top k for sampling"}
    )
    num_beams: int = field(
        default=1,
        metadata={"help": "Number of beams to use for sampling"}
    )
    length_penalty: float = field(
        default=0.0,
        metadata={"help": ""}
    )
    no_repeat_ngram_size: int = field(
        default=0,
        metadata={"help": ""}
    )
    min_tgt_length: int = field(
        default=0,
        metadata={"help": ""}
    )
    out_seq_legnth: int = field(
        default=256,
        metadata={"help": ""}
    )
    input_source: str = field(
        default="interactive",
        metadata={"help": "What input mode to use, interactive or path"}
    )
    output_path: str = field(
        default="./samples",
        metadata={"help": "Path to place the generated samples"}
    )
    with_id: bool = field(
        default=False,
        metadata={"help": "If each line is prepended with an id"}
    )
    max_inference_batch_size: int = field(
        default=12,
        metadata={"help": ""}
    )
    device: int = field(
        default=-1,
        metadata={"help": ""}
    )


@dataclass
class SwissTokenizationConfig(MetaseqDataclass):
    tokenization_type: str = field(
        default="fake",
        metadata={"help": "Type name of tokenizer"}
    )


@dataclass
class SwissConfig(MetaseqDataclass):
    """
    Similar structure to MetaseqConfig, but for facilitating use of
    the SwissArmyTransformer codebase. Carries a duplicate set of
    metaseq config args.
    """
    swiss_distributed_training: SwissDistributedTrainingConfig = SwissDistributedTrainingConfig()   # Note that this is the same as metaseq distributed config
    swiss_model: SwissModelConfig = SwissModelConfig()
    swiss_training: SwissTrainingConfig = SwissTrainingConfig()
    swiss_eval_lm: SwissEvaluationConfig = SwissEvaluationConfig()
    swiss_dataset: SwissDataConfig = SwissDataConfig()
    swiss_generation: SwissTextGenerationConfig = SwissTextGenerationConfig()
    swiss_tokenizer: SwissTokenizationConfig = SwissTokenizationConfig()
