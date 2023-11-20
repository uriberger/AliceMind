#!/bin/sh

venv/bin/python vqa_mplug.py \
    --config ./configs/vqa_mplug_large_style.yaml \
    --output_dir output/vqa_mplug_large_style \
    --checkpoint ./mplug_large.pth \
    --do_two_optim \
    --add_object \
    --max_input_length 80 \
    --do_amp \
    --add_ocr \
    --deepspeed \
    --deepspeed_config ./configs/ds_config.json
