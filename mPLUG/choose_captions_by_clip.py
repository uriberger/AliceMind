import torch
import clip
from PIL import Image
import json
import sys
import time

assert len(sys.argv) == 2
image_ids_file = sys.argv[1]

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

with open('../../CLIP_prefix_caption/dataset_coco.json', 'r') as fp:
    coco_data = json.load(fp)['images']

with open(image_ids_file, 'r') as fp:
    image_ids = json.load(fp)
image_ids_dict = {x: True for x in image_ids}

data = [x for x in coco_data if x['cocoid'] in image_ids_dict]
image_id_to_caption_inds = {}
count = 0
t = time.time()
for sample in data:
    if count % 1000 == 0:
        print('Starting sample ' + str(count) + ' out of ' + str(len(data)) + ', time from prev ' + str(time.time() - t), flush=True)
        t = time.time()
    image_id = sample['cocoid']
    image_path = '/cs/labs/oabend/uriber/datasets/COCO/val2014/COCO_val2014_' + str(image_id).zfill(12) + '.jpg'
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text = clip.tokenize([x['raw'] for x in sample['sentences']]).to(device)

    with torch.no_grad():
        logits_per_image, logits_per_text = model(image, text)
        res_list = logits_per_image[0].tolist()
        res_list = [(i, res_list[i]) for i in range(len(res_list))]
        res_list.sort(key=lambda x:x[1], reverse=True)
        caption_inds = [x[0] for x in res_list[:2]]
        image_id_to_caption_inds[image_id] = caption_inds
    count += 1

with open('res.json', 'w') as fp:
    fp.write(json.dumps(image_id_to_caption_inds))
print('Finished!')
