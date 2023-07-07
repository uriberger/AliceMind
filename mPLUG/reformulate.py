import sys
import os
import yaml
import json
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from models.model_vqa_mplug import MPLUG
from models.tokenization_bert import BertTokenizer
from models.vit import resize_pos_embed

assert len(sys.argv) == 3

model_path = sys.argv[1]
with open(sys.argv[2], 'r') as fp:
    data = json.load(fp)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
config = yaml.load(open('configs/vqa_mplug_base.yaml', 'r'), Loader=yaml.Loader)
config["min_length"] = 1
config["max_length"] = 50
config["beam_size"] = 5
config['add_ocr'] = False
config['add_object'] = False
config['text_encoder'] = 'bert-base-uncased'
config['text_decoder'] = 'bert-base-uncased'

print("Creating model")
model = MPLUG(config=config, tokenizer=tokenizer)
checkpoint = torch.load(model_path, map_location='cpu')
state_dict = checkpoint['model']
model.load_state_dict(state_dict, strict=False)
model.eval()
device = torch.device('cuda')
model = model.to(device)
coco_root = config['coco_root']

normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
transform = transforms.Compose([
    transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
    transforms.ToTensor(),
    normalize,
    ])

res = []
for sample in data:
    image_id = sample['image_id']
    image_path = os.path.join(coco_root, 'val2014', 'COCO_val2014_' + str(image_id).zfill(12) + '.jpg')
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.to(device, non_blocking=True)
    image = image.unsqueeze(0)

    question = sample['caption']
    question_input = tokenizer(question, padding='longest', return_tensors="pt").to(device)
    topk_ids, topk_probs = model(image, question_input, answer=None, train=False, k=config['k_test'])
    ans = tokenizer.decode(topk_ids[0][0]).replace("[SEP]", "").replace("[CLS]", "").replace("[PAD]", "").strip()
    res.append({'cocoid': image_id, 'sentences': [{'raw': ans}], 'filepath': 'val2014', 'filename': 'COCO_val2014_' + str(image_id).zfill(12) + '.jpg'})

with open('ann.json', 'w') as fp:
    fp.write(json.dumps(res))
