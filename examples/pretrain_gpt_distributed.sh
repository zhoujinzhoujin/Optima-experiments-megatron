#! /bin/bash

# Runs the "345M" parameter model

GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=/home/ubuntu/Megatron_dataset/gpt2-345m/my-gpt2_text_document
# CHECKPOINT_PATH=/workspace/checkpoints/gpt2-345m/

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
# medium: 24 1024 16
# 1b: 32 1536 24
# 4b: 32 3072 24
# --distribute-saved-activations \
torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
       --distributed-backend nccl \
       --tensor-model-parallel-size $GPUS_PER_NODE \
       --pipeline-model-parallel-size 1 \
       --recompute-activations \
       --recompute-granularity full \
       --seed 123 \
       --num-layers 48 \
       --hidden-size 3072 \
       --num-attention-heads 24 \
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
       --vocab-file /home/ubuntu/Megatron_dataset/gpt2-345m/gpt2-vocab.json \
       --merge-file /home/ubuntu/Megatron_dataset/gpt2-345m/gpt2-merges.txt
#       --load $CHECKPOINT_PATH \
