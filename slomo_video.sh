#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python slomo_video.py \
    --input_video_path videos/dog.mp4 \
    --output_video_path videos/dog_slomo.mp4 \
    --slomo_rate 10 \
    --checkpoint /mnt/069A453E9A452B8D/Ram/slomo_data/experiment_lrelu/model-200000