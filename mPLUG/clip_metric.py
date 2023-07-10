import torch
import clip
from PIL import Image
import json
import sys
import os
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

dataset_name = sys.argv[1]
if dataset_name == 'COCO':
    image_id_to_path = lambda image_id: '/cs/labs/oabend/uriber/datasets/COCO/val2014/COCO_val2014_' + str(image_id).zfill(12) + '.jpg'
elif dataset_name == 'pascal':
    images_root = '/cs/labs/oabend/uriber/datasets/pascal_sentences/dataset'
    image_id_to_path_dict = {}
    for dir_name in os.listdir(images_root):
        dir_path = os.path.join(images_root, dir_name)
        if not os.path.isdir(dir_path):
            continue
        for file_name in os.listdir(dir_path):
            image_id = int(file_name.split('2008_')[1].split('.jpg')[0])
            image_id_to_path_dict[image_id] = os.path.join(dir_path, file_name)
    image_id_to_path = lambda image_id: image_id_to_path_dict[image_id]
else:
    assert False, 'Unknown dataset ' + dataset_name

input_files = sys.argv[2:]
data = {}
for input_file in input_files:
    with open(input_file, 'r') as fp:
        data[input_file] = json.load(fp)
'''with open('reformulation_data/test_ids.json', 'r') as fp:
    image_ids = json.load(fp)
image_ids_dict = {x: True for x in image_ids}'''
first_file_name = input_files[0]

res = {x: 0 for x in input_files}
sample_num = len(data[first_file_name])

t = time.time()
for i in range(sample_num):
    if i % 1000 == 0:
        print('Staring sample ' + str(i) + ' out of ' + str(sample_num) + ', time from prev ' + str(time.time() - t), flush=True)
        t = time.time()
    image_id = data[first_file_name][i]['image_id']
    '''if image_id not in image_ids_dict:
        continue'''
    for input_file in input_files[1:]:
        assert image_id == data[input_file][i]['image_id']
    image_path = image_id_to_path(image_id)
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text = clip.tokenize([data[input_file][i]['caption'] for input_file in input_files]).to(device)

    with torch.no_grad():
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

        for j in range(probs.shape[0]):
            res[input_files[j]] += probs[j]

for input_file in input_files:
    print(input_file + ': ' + str(res[input_file]/sample_num))
