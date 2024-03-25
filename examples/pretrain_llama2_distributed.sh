#! /bin/bash

# Runs the "345M" parameter model

GPUS_PER_NODE=1
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=/home/ubuntu/Megatron_dataset/llama2-7b/my-llama2_text_document

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
# 32
torchrun $DISTRIBUTED_ARGS pretrain_llama2.py \
       --distributed-backend nccl \
       --tensor-model-parallel-size $GPUS_PER_NODE \
       --pipeline-model-parallel-size 1 \
       --recompute-activations \
       --recompute-granularity full \
       --seed 123 \
       --num-layers 28 \
       --hidden-size 4096 \
       --ffn-hidden-size 11008 \
       --num-attention-heads 32 \
       --micro-batch-size 1 \
       --seq-length 2048 \
       --max-position-embeddings 4096 \
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
       --vocab-file /home/ubuntu/Megatron_dataset/llama2-7b/gpt2-vocab.json \
       --merge-file /home/ubuntu/Megatron_dataset/llama2-7b/gpt2-merges.txt \
       --tokenizer-name-or-path meta-llama/Llama-2-7b-hf \
       --tokenizer-not-use-fast
#       --load $CHECKPOINT_PATH \