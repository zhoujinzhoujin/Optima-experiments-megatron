#! /bin/bash

# Runs the "345M" parameter model

GPUS_PER_NODE=1
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=/workspace/dataset/gpt2-345m/my-gpt2_text_document
# CHECKPOINT_PATH=/workspace/checkpoints/gpt2-345m/

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
       --distributed-backend nccl \
       --tensor-model-parallel-size $GPUS_PER_NODE \
       --pipeline-model-parallel-size 1 \
       --seed 123 \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --micro-batch-size 1 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 10 \
       --lr-decay-iters 320000 \
       --data-path $DATA_PATH \
       --data-impl mmap \
       --split 1,0,0 \
       --lr 1.0e-3 \
       --weight-decay 0.0 \
       --clip-grad 1.0e12 \
       --eval-iters 0 \
       --log-interval 1 \
       --vocab-file /workspace/dataset/gpt2-345m/gpt2-vocab.json \
       --merge-file /workspace/dataset/gpt2-345m/gpt2-merges.txt
#       --load $CHECKPOINT_PATH \
