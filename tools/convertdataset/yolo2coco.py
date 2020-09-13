import json
import os
from os import walk,listdir
import cv2
import numpy as np
# from sympy import symbols, Eq, solve
import tqdm
import pickle
from multiprocessing import Manager, Pool
from functools import partial
import argparse

train_img = '../data/images/full/full/'
train_anno = train_img
save_anno = './cache/first.json'

def parse_args():
    parser = argparse.ArgumentParser(description='Convert YOLO json file to COCO json file')
    parser.add_argument('img_dir', help='path to image')
    parser.add_argument('anno_dir', help='path to anno fils')
    parser.add_argument('save_anno_file', help='path and name to save coco json file after converting')
    args = parser.parse_args()
    return args

args = parse_args()
train_img = args.img_dir
train_anno = args.anno_dir
save_anno = args.save_anno_file

def getbox(path, H, W):
    with open(path) as f:
        data = np.loadtxt(f)
    if data.ndim == 1:
        data = data.reshape(1,-1)
    if data.size == 0:
        return 0
    bboxs = []
    data[:,1] *= W
    data[:,3] *= W 
    data[:,2] *= H
    data[:,4] *= H 

    for box in data:
        bboxs.append(box)
    return bboxs

def make_annotation(img_path, anno_path, an_format='.txt'):
    
    images = []
    annotations = []
    img_id = 0 #imgae id
    ann_id = 9999 # annotation id
    count = 1 # for debug

    # go through all annotations and files
    for file in tqdm.tqdm(os.listdir(img_path)):
        name, extension = os.path.splitext(file)
        if extension.lower() != '.txt' or name=='classes':
            continue
        if os.path.exists(os.path.join(img_path,name+'.JPG')) == False:
            print('file is not existed: ', os.path.join(img_path,name+'.JPG'))
            continue

        img_id = img_id + 1
        file = name + '.JPG'
        img = cv2.imread(os.path.join(img_path,file))
        H,W,_ = img.shape

        txtfile = os.path.join(anno_path, name + an_format)
        boxes = getbox(txtfile, H=H, W=W)
        
        # skip empty annotaion
        if isinstance(boxes,int):
            print('file is empty: ', txtfile)
            continue
        
        images.append({"date_captured" : "None",
                            "file_name" : file,
                            "id" : img_id,
                            "license" : 1,
                            "url" : "",
                            "height" : H,
                            "width" : W})
        for box in boxes:
            x,y,w,h = box[1:].astype(int)
            xt,yt = x - w/2, y - h/2
            # xb,yb = x + w/2, y + h/2
            ann_id = ann_id + 1
            annotations.append({"segmentation" : 1,
                                    "area" : 1,
                                    "iscrowd" : 0,
                                    "image_id" : img_id,
                                    "bbox" : [float(xt), float(yt),float(w),float(h)],
                                    "category_id" : int(box[0]+1),
                                    "id": ann_id})
            if count < 10:
                img = cv2.rectangle(img, (x-w//2,y-h//2) , (x+w//2, y+h//2), color = (255, 0, 0) , thickness=2) 


        if count < 10:
            cv2.imwrite(f'./cache/img{count}.jpg', img)
        # if count >= 5:
        #     break
        count = count + 1
    return images, annotations
        
if __name__=='__main__':

    categories = [{'id': 1, 'name': 'motorbike'}, {'id': 2, 'name': 'car'},{'id': 3, 'name': 'bus'}, {'id': 4, 'name': 'truck'},{'id': 5, 'name': 'person'}]

    # # train annotation 
    images,annotations = make_annotation(train_img, train_anno)
    data = {'images':images,'annotations':annotations,'categories':categories,'licenses':1, 'info':1}
    # import ipdb; ipdb.set_trace()
    with open(save_anno + 'thesis_train.json', 'w') as json_file:
        json.dump(data, json_file)
    print('finish train annotation')