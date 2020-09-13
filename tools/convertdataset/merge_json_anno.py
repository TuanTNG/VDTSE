import os
import argparse
import tqdm
import json


ann_dir = '/data/Toda/data_for_train/knot/annotations/'
jsonfiles = ['v1_train.json', 'v2_train.json']  # list of merge files
save_path = ann_dir + 'v12_train.json'


def parse_args():
    parser = argparse.ArgumentParser(
        description='merge 2 json files, COCO format')
    parser.add_argument(
        '--ann_dir', help='prefix to annotation directory', default=ann_dir)
    parser.add_argument(
        '--jsonfiles', default=jsonfiles, help='list of json files to be merged', nargs='+', type=str)
    parser.add_argument(
        '--save_path', default=save_path, help='path to save merged json file')

    args = parser.parse_args()
    return args


args = parse_args()
ann_dir = args.ann_dir
jsonfiles = args.jsonfiles
save_path = args.save_path

print("Merging annotations")

cann = {'images': [],
        'annotations': [],
        'info': None,
        'licenses': None,
        'categories': None}


print("Merging annotations")

for j in tqdm.tqdm(jsonfiles):
    with open(os.path.join(ann_dir, j)) as a:
        cj = json.load(a)

    ind = jsonfiles.index(j)
    # Check if this is the 1st annotation.
    # If it is, continue else modify current annotation
    if ind == 0:
        cann['images'] = cann['images'] + cj['images']
        cann['annotations'] = cann['annotations'] + cj['annotations']
        if 'info' in list(cj.keys()):
            cann['info'] = cj['info']
        if 'licenses' in list(cj.keys()):
            cann['licenses'] = cj['licenses']
        cann['categories'] = cj['categories']

        last_imid = cann['images'][-1]['id']
        last_annid = cann['annotations'][-1]['id']

        # If last imid or last_annid is a str, convert it to int
        if isinstance(last_imid, str) or isinstance(last_annid, str):
            logging.debug("String Ids detected. Converting to int")
            id_dict = {}
            # Change image id in images field
            for i, im in enumerate(cann['images']):
                id_dict[im['id']] = i
                im['id'] = i

            # Change annotation id & image id in annotations field
            for i, im in enumerate(cann['annotations']):
                im['id'] = i
                if isinstance(last_imid, str):
                    im['image_id'] = id_dict[im['image_id']]

        last_imid = cann['images'][-1]['id']
        last_annid = cann['annotations'][-1]['id']

    else:

        id_dict = {}
        # Change image id in images field
        for i, im in enumerate(cj['images']):
            id_dict[im['id']] = last_imid + i + 1
            im['id'] = last_imid + i + 1

        # Change annotation and image ids in annotations field
        for i, ann in enumerate(cj['annotations']):
            ann['id'] = last_annid + i + 1
            ann['image_id'] = id_dict[ann['image_id']]

        cann['images'] = cann['images'] + cj['images']
        cann['annotations'] = cann['annotations'] + cj['annotations']
        if 'info' in list(cj.keys()):
            cann['info'] = cj['info']
        if 'licenses' in list(cj.keys()):
            cann['licenses'] = cj['licenses']
        cann['categories'] = cj['categories']

        last_imid = cann['images'][-1]['id']
        last_annid = cann['annotations'][-1]['id']

with open(save_path, 'w') as aw:
    json.dump(cann, aw)
print('merged file is save at: ', save_path)