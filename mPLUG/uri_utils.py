def reshape_output_list(output_list):
    return [{'image_id': int(x['question_id'].split('2014_')[1].split('.jpg')[0]), 'caption': x['pred_caption']} for x in output_list]
