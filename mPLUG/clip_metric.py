import torch
import clip
from PIL import Image
import json
from uri_utils import reshape_output_list
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

with open('output/after_batch_5_reformulations/result/vqa_result_epoch0.json', 'r') as fp:
    re_data = json.load(fp)
    re_data = reshape_output_list(re_data)
with open('output/after_batch_5_gt/result/vqa_result_epoch0.json', 'r') as fp:
    gt_data = json.load(fp)
    gt_data = reshape_output_list(gt_data)
with open('output/after_batch_5_gt2/result/vqa_result_epoch0.json', 'r') as fp:
    gt2_data = json.load(fp)
    gt2_data = reshape_output_list(gt2_data)
with open('output/after_batch_5_gt_clip/result/vqa_result_epoch0.json', 'r') as fp:
    clip_data = json.load(fp)
    clip_data = reshape_output_list(clip_data)
with open('reformulation_data/test_ids.json', 'r') as fp:
    image_ids = json.load(fp)
image_ids_dict = {x: True for x in image_ids}

res = [0]*4

t = time.time()
for i in range(len(re_data)):
    if i % 100 == 0:
        print('Staring sample ' + str(i) + ' out of ' + str(len(re_data)) + ', time from prev ' + str(time.time() - t), flush=True)
        t = time.time()
    image_id = re_data[i]['image_id']
    if image_id not in image_ids_dict:
        continue
    assert image_id == gt_data[i]['image_id'] and image_id == gt2_data[i]['image_id'] and image_id == clip_data[i]['image_id']
    image_path = '/cs/labs/oabend/uriber/datasets/COCO/val2014/COCO_val2014_' + str(image_id).zfill(12) + '.jpg'
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    text = clip.tokenize([re_data[i]['caption'], gt_data[i]['caption'], gt2_data[i]['caption'], clip_data[i]['caption']]).to(device)

    with torch.no_grad():
        logits_per_image, logits_per_text = model(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()[0]

        for j in range(probs.shape[0]):
            res[j] += probs[j]

res_names = ['re', 'gt', 'gt2', 'clip']
for i in range(len(res_names)):
    print(res_names[i] + ': ' + str(res[i]/len(image_ids)))
