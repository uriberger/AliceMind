import torch
import os
import math
import time
import json
import yaml
import argparse
from torchvision import transforms
from PIL import Image
import torch.nn as nn
from models.model_caption_mplug import MPLUG
from models.tokenization_bert import BertTokenizer
from models.visual_transformers import resize_pos_embed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_id_file', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--dataset', default='COCO')
    args = parser.parse_args()
        
    with open(args.image_id_file, 'r') as fp:
        image_ids = json.load(fp)

    config = yaml.load(open('configs/caption_mplug_large.yaml', 'r'), Loader=yaml.Loader)
    config['text_encoder'] = 'bert-base-uncased'
    config['text_decoder'] = 'bert-base-uncased'
    config['beam_size'] = 5
    config['min_length'] = 8
    config['max_length'] = 25

    if args.dataset == 'COCO':
        with open('../../CLIP_prefix_caption/dataset_coco.json', 'r') as fp:
            coco_data = json.load(fp)['images']
        image_id_to_split = {}
        for x in coco_data:
            if x['split'] == 'train':
                split = 'train'
            else:
                split = 'val'
            image_id_to_split[x['cocoid']] = split
        image_id_to_path = lambda x: f'{config["coco_root"]}/{image_id_to_split[x]}2014/COCO_{image_id_to_split[x]}2014_{str(x).zfill(12)}.jpg'
    elif args.dataset == 'flickr30k':
        image_id_to_path = lambda x: f'{config["flickr30k_root"]}/images/{x}.jpg'
    elif args.dataset == 'aic':
        image_id_to_split = {}
        with open('/cs/labs/oabend/uriber/datasets/ai_challenger/ai_challenger_caption_train_20170902/caption_train_annotations_20170902.json', 'r') as fp:
            aic_train_data = json.load(fp)
        with open('/cs/labs/oabend/uriber/datasets/ai_challenger/ai_challenger_caption_validation_20170910/caption_validation_annotations_20170910.json', 'r') as fp:
            aic_val_data = json.load(fp)
        for sample in aic_train_data:
            image_id = int(sample['image_id'].split('.jpg')[0], 16)
            image_id_to_split[image_id] = 'train'
        for sample in aic_val_data:
            image_id = int(sample['image_id'].split('.jpg')[0], 16)
            image_id_to_split[image_id] = 'validation'
        split_to_date = {'train': '20170902', 'validation': '20170910'}
        image_id_to_path = lambda x: f'{config["aic_root"]}/ai_challenger_caption_{image_id_to_split[x]}_{split_to_date[image_id_to_split[x]]}/caption_{image_id_to_split[x]}_images_{split_to_date[image_id_to_split[x]]}/{hex(x)[2:].zfill(40)}.jpg'
    else:
        assert False, f'Unkown dataset {args.dataset}'

    print("Loading model")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = MPLUG(config=config, tokenizer=tokenizer)
    checkpoint = torch.load(args.model_path, map_location='cpu')
    state_dict = checkpoint['model']

    num_patches = int(config["image_res"] * config["image_res"]/(14*14))
    pos_embed = nn.Parameter(torch.zeros(num_patches + 1, 1024).float())

    #pos_embed = resize_pos_embed(state_dict['visual_encoder.visual.positional_embedding'].unsqueeze(0), pos_embed.unsqueeze(0))
    state_dict['visual_encoder.visual.positional_embedding'] = pos_embed

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    device = torch.device('cuda')
    model = model.to(device)
    print('Model loaded')

    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    transform = transforms.Compose([
        transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])

    if os.path.isfile(args.output_file):
        with open(args.output_file, 'r') as fp:
            res = json.load(fp)
    else:
        res = []
    batch_start = len(res)
    batch_ind = 0
    batch_num = math.ceil((len(image_ids)-len(res))/args.batch_size)
    t = time.time()
    while batch_start < len(image_ids):
        if batch_ind % 100 == 0:
            print(f'Starting batch {batch_ind} out of {batch_num}, time from prev {time.time() - t}', flush=True)
            t = time.time()
            with open(args.output_file, 'w') as fp:
                fp.write(json.dumps(res))
        batch_end = min(batch_start + args.batch_size, len(image_ids))
        batch_inds = [i for i in range(batch_start, batch_end)]

        questions = [config['bos']+" " for _ in range(batch_end-batch_start)]
        question_input = tokenizer(questions, padding='longest', return_tensors="pt").to(device)

        batch_image_ids = [image_ids[i] for i in batch_inds]
        batch_image_paths = [image_id_to_path(x) for x in batch_image_ids]
        images = torch.cat([transform(Image.open(image_path).convert('RGB')).unsqueeze(0) for image_path in batch_image_paths], dim=0).to(device, non_blocking=True)

        topk_ids, topk_probs = model(images, question_input, answer=None, train=False)
        answers = [tokenizer.decode(topk_ids[i][0]).replace("[SEP]", "").replace("[CLS]", "").replace("[PAD]", "").strip() for i in range(len(topk_ids))]
        res += [{'image_id': batch_image_ids[i], 'caption': answers[i]} for i in range(len(batch_inds))]

        batch_start = batch_end
        batch_ind += 1

    with open(args.output_file, 'w') as fp:
        fp.write(json.dumps(res))
    print('Finished!')
