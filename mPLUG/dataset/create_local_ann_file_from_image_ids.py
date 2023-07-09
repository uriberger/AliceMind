import argparse
import json
import os
import random
from uri_utils import reshape_output_list

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_ind', type=int)
    parser.add_argument('--data_dir', default='reformulation_data')
    parser.add_argument('--select_caption_method', default='random')
    parser.add_argument('--clip_image_id_to_caption_inds_file')
    parser.add_argument('--data_file')
    args = parser.parse_args()

    batch_ind = args.batch_ind
    image_ids_file = os.path.join(args.data_dir, 'batch_' + str(batch_ind), 'batch_' + str(batch_ind) + '_image_ids.json')
    with open(image_ids_file, 'r') as fp:
        image_id_list = json.load(fp)
    select_caption_method = args.select_caption_method
    assert select_caption_method in ['random', 'clip'], 'Unknown select caption method ' + str

    if select_caption_method == 'clip':
        if args.clip_image_id_to_caption_inds_file is None:
            image_id_to_caption_inds_file = 'reformulation_data/batch_' + str(batch_ind) + '/image_id_to_caption_inds.json'
        else:
            image_id_to_caption_inds_file = args.image_id_to_caption_inds_file
        with open(image_id_to_caption_inds_file, 'r') as fp:
            image_id_to_caption_inds = json.load(fp)
    
    ann_data = []

    if args.data_file is None:
        with open('../../CLIP_prefix_caption/dataset_coco.json', 'r') as fp:
            all_coco_data = json.load(fp)['images']
    else:
        with open(args.data_file, 'r') as fp:
            all_coco_data = json.load(fp)
            all_coco_data = reshape_output_list(all_coco_data)

    for image_id in image_id_list:
        sample = [x for x in all_coco_data if x['cocoid'] == image_id][0]
        if select_caption_method == 'random':
            caption_inds = random.sample(range(len(sample['sentences'])), 2)
        elif select_caption_method == 'clip':
            caption_inds = image_id_to_caption_inds[str(image_id)]
        res = sample
        res['sentences'] = [res['sentences'][i] for i in caption_inds]
        ann_data.append(res)

    with open('ann.json', 'w') as fp:
        fp.write(json.dumps(ann_data))

    print('Finished, loaded ' + str(len(ann_data)) + ' images with ' + str(sum([len(x['sentences']) for x in ann_data])) + ' captions')
