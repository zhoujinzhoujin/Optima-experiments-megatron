#! /bin/bash

# Runs the "345M" parameter model

RANK=0
WORLD_SIZE=1

DATA_PATH=/workspace/dataset/gpt2-345m/my-gpt2_text_document
CHECKPOINT_PATH=/workspace/checkpoints/gpt2-345m/


torchrun pretrain_gpt.py \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --micro-batch-size 1 \
       --global-batch-size 1 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 10 \
       --lr-decay-iters 320000 \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --data-impl mmap \
       --split 1,0,0 \
       --distributed-backend nccl \
       --lr 0.0001 \
       --min-lr 1.0e-5 \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --eval-iters 0 \
       --fp16 \
       --seed 123 \
       --log-interval 1 \
      --vocab-file /workspace/dataset/gpt2-345m/gpt2-vocab.json \
      --merge-file /workspace/dataset/gpt2-345m/gpt2-merges.txt