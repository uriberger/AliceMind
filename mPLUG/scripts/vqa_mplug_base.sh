#!/bin/sh
# deepspeed==0.5.8

venv/bin/python vqa_mplug.py \
    --config ./configs/vqa_mplug_base.yaml \
    --checkpoint ./mplug_base.pth \
    --output_dir output/vqa_mplug_base \
    --do_two_optim \
    --add_object \
    --max_input_length 80 \
    --max_length 50 \
    --add_ocr
