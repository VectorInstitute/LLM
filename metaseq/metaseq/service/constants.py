# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

MAX_SEQ_LEN = 2048
BATCH_SIZE = 2048  # silly high bc we dynamically batch by MAX_BATCH_TOKENS
MAX_BATCH_TOKENS = 3072
DEFAULT_PORT = 6010
MAX_BEAM = 16

# 175B
#MODEL_PARALLEL = 32
#TOTAL_WORLD_SIZE = 32

# Test 125M
MODEL_PARALLEL = 2
TOTAL_WORLD_SIZE = 2

try:
    # internal logic denoting where checkpoints are in meta infrastructure
    from metaseq_internal.constants import LOCAL_SSD, MODEL_SHARED_FOLDER 
except ImportError: # CHECKPOINT_FOLDER should point to a shared drive (e.g. NFS) where the checkpoints from S3 are stored. As an example:
    # CHECKPOINT_FOLDER = "/example/175B/reshard_no_os"
    # $ ls /example/175B/reshard_no_os
    # reshard-model_part-0.pt
    # reshard-model_part-1.pt
    # reshard-model_part-2.pt
    # reshard-model_part-3.pt
    # reshard-model_part-4.pt
    # reshard-model_part-5.pt
    # reshard-model_part-6.pt
    # reshard-model_part-7.pt
    #CHECKPOINT_FOLDER = "/example/175B/reshard_no_os"
    MODEL_SHARED_FOLDER = "/checkpoint/opt_test/original"
    #MODEL_SHARED_FOLDER = "/scratch/ssd001/jba/OPT"

    # Choose which model to use
    MODEL_FOLDER = os.path.join(MODEL_SHARED_FOLDER, "OPT-125M")

# tokenizer files
BPE_MERGES = os.path.join(MODEL_SHARED_FOLDER, "gpt2-merges.txt")
BPE_VOCAB = os.path.join(MODEL_SHARED_FOLDER, "gpt2-vocab.json")

CHECKPOINT_FOLDER = MODEL_FOLDER #os.path.join(MODEL_FOLDER, "reshard_no_os")

CHECKPOINT_LOCAL = os.path.join(MODEL_FOLDER, "reshard_no_os", "reshard.pt")

LAUNCH_ARGS = [
    # TODO: Testing swiss
    "--arch glm_large",
    "--memory-efficient-fp16",
    f"--model-parallel-size {MODEL_PARALLEL}",
    f"--distributed-world-size {TOTAL_WORLD_SIZE}",
    "--task language_modeling",
    f"--bpe-merges {BPE_MERGES}",
    f"--bpe-vocab {BPE_VOCAB}",
    "--bpe hf_byte_bpe",
    f"--merges-filename {BPE_MERGES}",  # TODO(susanz): hack for getting interactive_hosted working on public repo
    f"--vocab-filename {BPE_VOCAB}",  # TODO(susanz): hack for getting interactive_hosted working on public repo
    f"--path {CHECKPOINT_FOLDER}/reshard.pt",
    "--beam 1 --nbest 1",
    "--distributed-port 13000",
    "--checkpoint-shard-count 1",
    "--use-sharded-state",
    f"--batch-size {BATCH_SIZE}",
    f"--buffer-size {BATCH_SIZE * MAX_SEQ_LEN}",
    f"--max-tokens {BATCH_SIZE * MAX_SEQ_LEN}",
    "/tmp",  # required "data" argument.
]
