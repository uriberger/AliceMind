import argparse
import math
import os
import time
import yaml
import json
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from models.model_vqa_mplug import MPLUG
from models.tokenization_bert import BertTokenizer
from models.vit import resize_pos_embed

def remove_long_samples(input_ids):
    inds_to_remove = []
    for i in range(input_ids.shape[0]):
        if input_ids[i, 512].item() != 0:
            inds_to_remove.append(i)
    return inds_to_remove

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split')
    parser.add_argument('--dataset', default='COCO', choices=['COCO', 'flickr30k', 'aic', 'xm3600', 'ImageNet'])
    parser.add_argument('--output_format', default='image')
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--input_file', required=True)
    parser.add_argument('--output_file', required=True)
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    if args.dataset == 'COCO':
        split = args.split
        image_id_to_split = {}
        with open('../../CLIP_prefix_caption/dataset_coco.json', 'r') as fp:
            coco_data = json.load(fp)['images']
        for sample in coco_data:
            image_id = sample['cocoid']
            if split is None:
                image_id_to_split[image_id] = sample['filepath'].split('2014')[0]
            else:
                image_id_to_split[image_id] = split
    elif args.dataset == 'aic':
        split = args.split
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
            
    batch_size = args.batch_size
    output_file_name = args.output_file + '.json'

    model_path = args.model_path
    with open(args.input_file, 'r') as fp:
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
    flickr30k_root = config['flickr30k_root']
    aic_root = config['aic_root']
    xm3600_root = config['xm3600_root']
    image_net_root = config['image_net_root']

    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    transform = transforms.Compose([
        transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])

    if os.path.isfile(output_file_name):
        with open(output_file_name, 'r') as fp:
            res = json.load(fp)
    else:
        res = []
    data = data[len(res):]
    batch_start = 0
    batch_ind = 0
    batch_num = math.ceil(len(data)/batch_size)
    t = time.time()
    while batch_start < len(data):
        if batch_ind % 100 == 0:
            print(f'Starting batch {batch_ind} out of {batch_num}, time from prev {time.time() - t}', flush=True)
            t = time.time()
            with open(output_file_name, 'w') as fp:
                fp.write(json.dumps(res))
        batch_end = min(batch_start + batch_size, len(data))
        batch_inds = [i for i in range(batch_start, batch_end)]

        questions = [data[i]['caption'] for i in batch_inds]
        question_input = tokenizer(questions, padding='longest', return_tensors="pt").to(device)
        if question_input['input_ids'].shape[1] > 512:
            inds_to_remove = remove_long_samples(question_input['input_ids'])
            batch_inds = [i for i in batch_inds if i-batch_start not in inds_to_remove]
            questions = [data[i]['caption'] for i in batch_inds]
            question_input = tokenizer(questions, padding='longest', return_tensors="pt").to(device)

        image_ids = [data[i]['image_id'] for i in batch_inds]
        if args.dataset == 'COCO':
            image_paths = [os.path.join(coco_root, f'{image_id_to_split[image_id]}2014', f'COCO_{image_id_to_split[image_id]}2014_' + str(image_id).zfill(12) + '.jpg') for image_id in image_ids]
        elif args.dataset == 'flickr30k':
            image_paths = [os.path.join(flickr30k_root, 'images', f'{image_id}.jpg') for image_id in image_ids]
        elif args.dataset == 'aic':
            splits = [image_id_to_split[image_id] for image_id in image_ids]
            dates = [split_to_date[split] for split in splits]
            image_paths = [os.path.join(aic_root, f'ai_challenger_caption_{splits[i]}_{dates[i]}', f'caption_{splits[i]}_images_{dates[i]}', f'{hex(image_ids[i])[2:].zfill(40)}.jpg') for i in range(len(image_ids))]
        elif args.dataset == 'xm3600':
            image_paths = [os.path.join(xm3600_root, 'images', hex(image_id)[2:].zfill(16) + '.jpg') for image_id in image_ids]
        elif args.dataset == 'ImageNet':
            image_paths = [os.path.join(image_net_root, str(image_id) + '.jpg') for image_id in image_ids]
        images = torch.cat([transform(Image.open(image_path).convert('RGB')).unsqueeze(0) for image_path in image_paths], dim=0).to(device, non_blocking=True)

        topk_ids, topk_probs = model(images, question_input, answer=None, train=False, k=config['k_test'])
        answers = [tokenizer.decode(topk_ids[i][0]).replace("[SEP]", "").replace("[CLS]", "").replace("[PAD]", "").strip() for i in range(len(topk_ids))]
        if args.output_format == 'image':
            if args.dataset == 'COCO':
                res += [{'cocoid': data[batch_inds[i]]['image_id'], 'sentences': [{'raw': answers[i]}], 'filepath': f'{image_id_to_split[data[batch_inds[i]]["image_id"]]}2014', 'filename': f'COCO_{image_id_to_split[data[batch_inds[i]]["image_id"]]}2014_{str(data[batch_inds[i]]["image_id"]).zfill(12)}.jpg'} for i in range(len(batch_inds))]
            elif args.dataset == 'flickr30k':
                res += [{'cocoid': data[batch_inds[i]]['image_id'], 'sentences': [{'raw': answers[i]}], 'filepath': 'images', 'filename': f'{data[batch_inds[i]]["image_id"]}.jpg'} for i in range(len(batch_inds))]
        elif args.output_format == 'caption':
            res += [{'image_id': data[batch_inds[i]]['image_id'], 'caption': answers[i]} for i in range(len(batch_inds))]
        else:
            assert False

        batch_start = batch_end
        batch_ind += 1

    with open(output_file_name, 'w') as fp:
        fp.write(json.dumps(res))

    print('Finished!')
