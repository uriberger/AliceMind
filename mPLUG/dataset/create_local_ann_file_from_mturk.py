import sys
import csv
import json
from collections import defaultdict

assert len(sys.argv) == 2
mturk_results_file = sys.argv[1]
with open(mturk_results_file, 'r') as fp:
    my_reader = csv.reader(fp)
    data = []
    for row in my_reader:
         data.append(row)
data = data[1:]

ann_data = []

with open('../../CLIP_prefix_caption/dataset_coco.json', 'r') as fp:
    all_coco_data = json.load(fp)['images']

image_id_to_sentences = defaultdict(list)

for sample in data:
    if sample[29] != 'mplug':
        continue

    image_id = int(sample[27].split('2014_')[-1].split('.jpg')[0])
    caption = sample[30]
    image_id_to_sentences[image_id].append({'raw': caption})

for image_id, sentences in image_id_to_sentences.items():
    orig_sample = [x for x in all_coco_data if x['cocoid'] == image_id][0]
    res = sample
    res['sentences'] = sentences
    ann_data.append(res)

with open('ann.json', 'w') as fp:
    fp.write(json.dumps(ann_data))

print('Finished, loaded ' + str(len(ann_data)) + ' images with ' + str(sum([len(x['sentences']) for x in ann_data])) + ' captions')
