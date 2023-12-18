#!/bin/sh

venv/bin/python vqa_mplug.py \
    --config ./configs/vqa_mplug_large_style.yaml \
    --output_dir /cs/labs/oabend/uriber/romantic_reformulation_output \
    --checkpoint ./mplug_large.pth \
    --do_two_optim \
    --add_object \
    --max_input_length 80 \
    --add_ocr
