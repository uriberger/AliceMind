#!/bin/sh

CHECKPOINT=$1
OUTPUT_DIR=$2
echo "Checkpoint=$CHECKPOINT"
echo "Output dir=$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=0 venv/bin/python caption_mplug.py \
--config ./configs/caption_mplug_base.yaml \
--output_dir $CHECKPOINT \
--checkpoint $OUTPUT_DIR \
--text_encoder bert-base-uncased \
--text_decoder bert-base-uncased \
--min_length 8 \
--max_length 25 \
--max_input_length 25 \
--evaluate
