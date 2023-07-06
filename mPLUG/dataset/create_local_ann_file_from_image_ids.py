import sys
import json
import random
from uri_utils import reshape_output_list

assert len(sys.argv) == 2 or len(sys.argv) == 3 or len(sys.argv) == 4
batch_ind = int(sys.argv[1])
image_ids_file = 'reformulation_data/batch_' + str(batch_ind) + '/batch_' + str(batch_ind) + '_image_ids.json'
with open(image_ids_file, 'r') as fp:
    image_id_list = json.load(fp)
if len(sys.argv) > 2:
    select_caption_method = sys.argv[2]
    assert select_caption_method in ['random', 'clip'], 'Unknown select caption method ' + str
else:
    select_caption_method = 'random'

if select_caption_method == 'clip':
    with open('reformulation_data/batch_' + str(batch_ind) + '/image_id_to_caption_inds.json', 'r') as fp:
        image_id_to_caption_inds = json.load(fp)
    
ann_data = []

if len(sys.argv) > 3:
    with open(sys.argv[3], 'r') as fp:
        all_coco_data = json.load(fp)
        all_coco_data = reshape_output_list(all_coco_data)
else:
    with open('../../CLIP_prefix_caption/dataset_coco.json', 'r') as fp:
        all_coco_data = json.load(fp)['images']

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
