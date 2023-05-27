#!/bin/sh

for lr in 1e-5
do
    CUDA_VISIBLE_DEVICES=0 venv/bin/python caption_mplug.py \
    --config ./configs/caption_mplug_base.yaml \
    --output_dir output/coco_caption_base_$lr \
    --checkpoint ./mplug_base.pth \
    --text_encoder bert-base-uncased \
    --text_decoder bert-base-uncased \
    --do_two_optim \
    --lr $lr \
    --min_length 8 \
    --max_length 25 \
    --max_input_length 25
done
