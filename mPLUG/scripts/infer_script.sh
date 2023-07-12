#!/bin/sh

CHECKPOINT=$1
OUTPUT_DIR=$2
#CHECKPOINT=output/automatic/after_batch_20_reformulations/checkpoint_00.pth
#OUTPUT_DIR=auto_re
echo "Checkpoint=$CHECKPOINT"
echo "Output dir=$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=0 venv/bin/python caption_mplug.py \
--config ./configs/caption_mplug_base.yaml \
--output_dir $OUTPUT_DIR \
--checkpoint $CHECKPOINT \
--text_encoder bert-base-uncased \
--text_decoder bert-base-uncased \
--min_length 8 \
--max_length 25 \
--max_input_length 25 \
--evaluate
