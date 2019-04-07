#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --output_dir /mnt/069A453E9A452B8D/Ram/slomo_data/experiment_lrelu/ \
    --summary_dir /mnt/069A453E9A452B8D/Ram/slomo_data/experiment_lrelu/log/ \
    --mode train \
    --batch_size 4 \
    --vgg_ckpt /mnt/069A453E9A452B8D/Ram/KAIST/SRGAN_data/vgg_19.ckpt\
    --perceptual_mode VGG54 \
    --learning_rate 0.0001 \
    --decay_step 100000 \
    --decay_rate 0.1 \
    --stair True \
    --beta 0.9 \
    --max_iter 200000 \
    --train_data_count 116241
