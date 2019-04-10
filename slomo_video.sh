#!/usr/bin/env bash
CUDA_VISIBLE_DEVICES=0 python slomo_video.py \
    --input_video_path videos/drift.mp4 \
    --output_video_path videos/drift_slomo.mp4 \
    --slomo_rate 8 \
    --checkpoint /mnt/069A453E9A452B8D/Ram/slomo_data/experiment_lrelu/model-200000