import os, cv2
import numpy as np
from glob import glob
import torch, mmcv, argparse

from mmcv.runner import load_checkpoint
from mmcv.parallel import collate, scatter

from mmdet.apis.inference import LoadImage
from mmdet.datasets.pipelines import Compose

from mmdet.apis import init_detector
from mmdet.models import build_detector

import numpy as np
import matplotlib.pyplot as plt
import os, cv2, mmcv, torch, cvut
from mmdet.datasets import build_dataset
import os
from mmcv import Config
import tqdm

import tracker

#------------------------------------------------------------------------------
#  Utilization
#------------------------------------------------------------------------------

def inference_detector(model, data):
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result

def get_data(img, cfg, device):
    test_pipeline = Compose([LoadImage()] + cfg.test_pipeline[1:])
    data = dict(img=img)
    data = test_pipeline(data)
    data = scatter(collate([data], samples_per_gpu=1), [device])[0]
    return data

class_names = ('motorbike', 'car', 'bus', 'truck', 'person')

def draw_bboxes(image, bboxes, color=(0,255,0), thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX, font_size=0.5, font_thickness=2):
    for bbox in bboxes:
        cls, x, y, w, h, v = [int(ele) for ele in bbox]
        x1 = x - w // 2
        x2 = x1 + w
        y1 = y - h // 2
        y2 = y1 + h
        cv2.rectangle(image, (x1,y1), (x2,y2), color, thickness=thickness)
        text = class_names[cls]
        
        if v != -1:
            text = text + ' ' + str(v)
            
        textsize = cv2.getTextSize(text, font, font_size, font_thickness)[0]
        cv2.rectangle(image, (x1, y1 - textsize[1] - 4), (x1 + textsize[0] + 4, y1), (0, 0, 0), -1)
        cv2.putText(image, text, (x1 + 2, y1 - 2), font, font_size, color, font_thickness)
        

def draw_count(image, count, color=(0,255,0), thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX, font_size=0.5, font_thickness=2):
    texts = []
    sizes = []
    
    for name, c in zip(class_names, count):
        texts.append(name + ': ' + str(int(c)))
        textsize = cv2.getTextSize(texts[-1], font, font_size, font_thickness)[0]
        sizes.append(textsize)
    
    w = max((s[0] for s in sizes))
    h = sum((s[1] for s in sizes)) + 10
    
    cv2.rectangle(image, (0, image.shape[0]-h), (w, image.shape[0]), (0, 0, 0), -1)
    org = image.shape[0] - 2
    for text, size in zip(texts, sizes):
        cv2.putText(image, text, (2, org), font, font_size, color, font_thickness)
        org -= size[1] + 2

#------------------------------------------------------------------------------
#  ArgumentParser
#------------------------------------------------------------------------------
parser = argparse.ArgumentParser()

parser.add_argument("cfg", type=str, default=None,
                    help="Config file")

parser.add_argument("--ckpt", type=str, default=None,
                    help="Checkpoint file")

parser.add_argument("--det_thr", type=float, default=0.3,
                    help="Detection threshold")

parser.add_argument("--data_dir", type=str,
                    default="/data/coco/images/val2017/",
                    help="Data directory")

parser.add_argument("--out_dir", type=str, default='cache',
                    help="font_scale to draw bounding boxes")

parser.add_argument("--thickness", type=int, default=5,
                    help="thickness to draw bounding boxes")

parser.add_argument("--font_scale", type=int, default=4,
                    help="font_scale to draw bounding boxes")

parser.add_argument("--device", type=str, default='cuda',
                    help="cpu or gpu")

parser.add_argument("--video-name", type=str,
                    help="name of video")


args = parser.parse_args()


#------------------------------------------------------------------------------
#  Main
#------------------------------------------------------------------------------
if __name__ == "__main__":

    # Build model
    model = init_detector(args.cfg, args.ckpt, device=args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    # build dataset
    cfg = Config.fromfile(args.cfg)
    print(cfg)
    dataset = build_dataset(cfg.data.test)
    # Inference
    # for i, data in tqdm.tqdm(enumerate(dataset)):
    #     data = scatter(collate([data], samples_per_gpu=1), [args.device])[0]
    
    
    # img_file la duong dan toi hinh anh cua anh hoac la anh sau di doc len bang opencv (str hoac array)

    # load video
    video_name = args.video_name
    vid = cv2.VideoCapture(os.path.join(args.data_dir,video_name))
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    os.makedirs(args.out_dir, exist_ok=True)
    vw = cv2.VideoWriter(os.path.join(args.out_dir,video_name), fourcc, 30, (1920, 1080))
    # -------------------------------

    tracker.camera_info(1920, 1080, 52, 58.040, 7)
    
    t = 0
    while True:
        t += (1/30)
        # data = get_data(img_file, model.cfg, next(model.parameters()).device)
        # import ipdb; ipdb.set_trace()
        ret, image = vid.read()
        if image is None:
            break 

        data = get_data(image, model.cfg, next(model.parameters()).device)


        result = inference_detector(model, data)

        #test        
        # image = mmcv.imread(data['img_metas'][0][0]['filename'])    

        bboxes = np.vstack(result)
        labels = [                          
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(result)
        ]
        labels = np.concatenate(labels)

        score_thr = args.det_thr
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds].tolist()

        # import ipdb; ipdb.set_trace()
        # convert output nay sang dang cua code track
        x1, y1, x2, y2 = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3],
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        detection = np.ndarray(shape=bboxes.shape, dtype=np.int32)
        detection[:, 4] = labels
        detection[:, 0] = x
        detection[:, 1] = y
        detection[:, 2] = w
        detection[:, 3] = h

        track, count = tracker.track(t, detection)

        draw_bboxes(image, track, thickness=2, font_size=2, font_thickness=2)
        draw_count(image, count, thickness=2, font_size=2, font_thickness=2)   
        vw.write(image)
        
    vw.release()
    vid.release()
