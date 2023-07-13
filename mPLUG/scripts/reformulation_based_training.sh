#!/bin/sh
set -e

AUTO_DIR=automatic_reformulation_data
OUTPUT_DIR=output/automatic
CUDA_VISIBLE_DEVICES=0
CMD_PREFIX="venv/bin/python caption_mplug.py"
ARGS="--text_encoder bert-base-uncased \
    --text_decoder bert-base-uncased \
    --do_two_optim \
    --lr 1e-5 \
    --min_length 8 \
    --max_length 25 \
    --max_input_length 25 \
    --no_eval_tool"
MSG_PREFIX=[LOG_MSG]

for BATCH_IND in {1..20}
do
    echo "$MSG_PREFIX starting batch $BATCH_IND"
    if [[ "$BATCH_IND" -eq 1 ]];
    then
        PREV_BATCH_RE_MODEL=mplug_base.pth
        VANILLA_DIR=$OUTPUT_DIR/vanilla
        bash scripts/infer_script.sh $PREV_BATCH_RE_MODEL $VANILLA_DIR
        PREV_BATCH_RE_OUTPUT=${VANILLA_DIR}/result/vqa_result_epoch10.json
        PREV_BATCH_GT_MODEL=mplug_base.pth
        PREV_BATCH_GT2_MODEL=mplug_base.pth
        PREV_BATCH_CLIP_MODEL=mplug_base.pth
    else
        PREV_BATCH_DIR_PREFIX=$OUTPUT_DIR/after_batch_$(($BATCH_IND-1))
        PREV_BATCH_RE_MODEL=${PREV_BATCH_DIR_PREFIX}_reformulations/checkpoint_00.pth
        PREV_BATCH_RE_OUTPUT=${PREV_BATCH_DIR_PREFIX}_reformulations/result/vqa_result_epoch0.json
        PREV_BATCH_GT_MODEL=${PREV_BATCH_DIR_PREFIX}_gt/checkpoint_00.pth
        PREV_BATCH_GT2_MODEL=${PREV_BATCH_DIR_PREFIX}_gt2/checkpoint_00.pth
        PREV_BATCH_CLIP_MODEL=${PREV_BATCH_DIR_PREFIX}_clip/checkpoint_00.pth
    fi

    # Reformulations
    echo "$MSG_PREFIX reformulation training"
    INPUT_DATA=${AUTO_DIR}/batch_${BATCH_IND}/batch_${BATCH_IND}_input_data.json
    venv/bin/python reshape_and_filter.py $PREV_BATCH_RE_OUTPUT $AUTO_DIR/batch_${BATCH_IND}/batch_${BATCH_IND}_image_ids.json $INPUT_DATA
    venv/bin/python reformulate.py --model_path output/vqa_mplug_base/checkpoint_07.pth --input_file $INPUT_DATA
    mv ann.json ${AUTO_DIR}/batch_${BATCH_IND}/batch_${BATCH_IND}_reformulation_data.json
    $CMD_PREFIX --config $AUTO_DIR/batch_${BATCH_IND}/reformulation_config.yaml --output_dir $OUTPUT_DIR/after_batch_${BATCH_IND}_reformulations --checkpoint $PREV_BATCH_RE_MODEL $ARGS

    # gt
    echo "$MSG_PREFIX gt training"
    venv/bin/python dataset/create_local_ann_file_from_image_ids.py --batch_ind ${BATCH_IND} --data_dir $AUTO_DIR --captions_per_image 1
    mv ann.json $AUTO_DIR/batch_${BATCH_IND}/batch_${BATCH_IND}_gt_data.json
    $CMD_PREFIX --config $AUTO_DIR/batch_${BATCH_IND}/gt_config.yaml --output_dir $OUTPUT_DIR/after_batch_${BATCH_IND}_gt --checkpoint $PREV_BATCH_GT_MODEL $ARGS

    # gt2
    echo "$MSG_PREFIX gt2 training"
    venv/bin/python dataset/create_local_ann_file_from_image_ids.py --batch_ind ${BATCH_IND} --data_dir $AUTO_DIR --captions_per_image 1
    mv ann.json $AUTO_DIR/batch_${BATCH_IND}/batch_${BATCH_IND}_gt2_data.json
    $CMD_PREFIX --config $AUTO_DIR/batch_${BATCH_IND}/gt2_config.yaml --output_dir $OUTPUT_DIR/after_batch_${BATCH_IND}_gt2 --checkpoint $PREV_BATCH_GT2_MODEL $ARGS

    # clip
    echo "$MSG_PREFIX clip training"
    venv/bin/python dataset/create_local_ann_file_from_image_ids.py --batch_ind ${BATCH_IND} --select_caption_method clip --clip_image_id_to_caption_inds_file $AUTO_DIR/image_id_to_caption_inds.json --data_dir $AUTO_DIR --captions_per_image 1
    mv ann.json $AUTO_DIR/batch_${BATCH_IND}/batch_${BATCH_IND}_clip_data.json
    $CMD_PREFIX --config $AUTO_DIR/batch_${BATCH_IND}/clip_config.yaml --output_dir $OUTPUT_DIR/after_batch_${BATCH_IND}_clip --checkpoint $PREV_BATCH_CLIP_MODEL $ARGS

done

echo "$MSG_PREFIX DONE!"
