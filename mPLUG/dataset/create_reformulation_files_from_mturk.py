import sys
import csv
import json
import random

assert len(sys.argv) > 1
mturk_result_files = sys.argv[1:]
data = []
for mturk_result_file in mturk_result_files:
    with open(mturk_result_file, 'r') as fp:
        my_reader = csv.reader(fp)
        first = True
        for row in my_reader:
            if first:
                first = False
                continue
            if row[29] == 'mplug':
                data.append(row)

ann_data = []

for sample in data:
    res = {}
    image_id = int(sample[27].split('2014_')[-1].split('.jpg')[0])
    res['question_id'] = image_id
    res['image'] = 'val2014/COCO_val2014_' + str(image_id).zfill(12) + '.jpg'
    res['question'] = sample[28]
    res['answer'] = sample[30]
    ann_data.append(res)

# Split to 80% train 10% val 10% test
random.shuffle(ann_data)
train_data_num = int(0.8*len(ann_data))
val_data_num = int(0.1*len(ann_data))
train_data = ann_data[:train_data_num]
val_data = ann_data[train_data_num:train_data_num + val_data_num]
test_data = ann_data[train_data_num + val_data_num:]

with open('reformulation_data/reformulation_train.json', 'w') as fp:
    fp.write(json.dumps(train_data))
with open('reformulation_data/reformulation_val.json', 'w') as fp:
    fp.write(json.dumps(val_data))
with open('reformulation_data/reformulation_test.json', 'w') as fp:
    fp.write(json.dumps(test_data))

print('Finished, loaded ' + str(len(train_data)) + ' train samples, ' + str(len(val_data)) + ' val samples and ' + str(len(test_data)) + ' test samples')
