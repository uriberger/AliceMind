#!/bin/sh

echo "Checkpoint=$1"
echo "Output dir=$2"

CUDA_VISIBLE_DEVICES=0 venv/bin/python caption_mplug.py \
--config ./configs/caption_mplug_base.yaml \
--output_dir output/$2 \
--checkpoint $1 \
--text_encoder bert-base-uncased \
--text_decoder bert-base-uncased \
--min_length 8 \
--max_length 25 \
--max_input_length 25 \
--evaluate
