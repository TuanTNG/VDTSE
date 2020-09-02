import os
import json

jsonfile = '/home/cybercore/tank/Vehicles-Detection-Tracking-Speed-estimation-pytorch-mmdet/data/annotations/thesis_train.json'

imgsfile = '/home/cybercore/tank/Vehicles-Detection-Tracking-Speed-estimation-pytorch-mmdet/data/images'

with open(jsonfile) as f:
    data = json.load(f)

json_list_imgs = []

for _x in data['images']:
    json_list_imgs.append(_x['file_name'])

count = 0

for i, _file in enumerate(os.listdir(imgsfile)):
    if _file not in json_list_imgs:
        count += 1
        os.remove(os.path.join(imgsfile,_file))
        print('removed: ', os.path.join(imgsfile,_file))

print('total remove ', count)
