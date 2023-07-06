import sys
import json
from uri_utils import reshape_output_list

assert len(sys.argv) == 4

with open(sys.argv[1], 'r') as fp:
    data = json.load(fp)
    data = reshape_output_list(data)

with open(sys.argv[2], 'r') as fp:
    image_ids = json.load(fp)
image_ids_dict = {x: True for x in image_ids}

data = [x for x in data if x['image_id'] in image_ids_dict]
with open(sys.argv[3], 'w') as fp:
    fp.write(json.dumps(data))
