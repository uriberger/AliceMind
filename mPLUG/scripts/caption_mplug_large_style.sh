#!/bin/sh

for lr in 1e-5
do
    CUDA_VISIBLE_DEVICES=0 venv/bin/python caption_mplug.py \
    --config ./configs/caption_mplug_large_style.yaml \
    --output_dir output/style_caption_large \
    --checkpoint ./mplug_large_v2.pth \
    --do_two_optim \
    --lr $lr \
    --min_length 8 \
    --max_length 25 \
    --max_input_length 25 \
    --caption_dataset general
done
