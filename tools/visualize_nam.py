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
        cv2.putText(
            image, class_names[cls] + ' ' + str(v), (x1,y1-2),
            font, font_size, color, thickness=font_thickness)

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

parser.add_argument("--seg_thr", type=float, default=0.5,
                    help="Segmentation threshold")

parser.add_argument("--data_dir", type=str,
                    default="/data/coco/images/val2017/",
                    help="Data directory")
parser.add_argument("--out_dir", type=str, default='cache',
                    help="font_scale to draw bounding boxes")

parser.add_argument("--num_imgs", type=int, default=50,
                    help="Number of images for visualization")

parser.add_argument("--thickness", type=int, default=5,
                    help="thickness to draw bounding boxes")

parser.add_argument("--font_scale", type=int, default=4,
                    help="font_scale to draw bounding boxes")

parser.add_argument("--device", type=str, default='cuda',
                    help="cpu or gpu")


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
    vid = cv2.VideoCapture('./videos/IMG_0685.MOV')
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    vw = cv2.VideoWriter('out.avi', fourcc, 30, (1920, 1080))
    # -------------------------------

    tracker.camera_info(1920, 1080, 60, 58.040, 5)
    
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

        # test                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              
        # image = mmcv.imread(data['img_metas'][0][0]['filename'])    

        bboxes = np.vstack(result)
        labels = [                                                                                                                                                                                                                                                                                                                                          
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(result)
        ]
        labels = np.concatenate(labels)

        score_thr = 0.3
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
        # import ipdb; ipdb.set_trace()
        # img = model.show_result(data['img_metas'][0][0]['filename'], result, score_thr=args.det_thr, show=False,  thickness=args.thickness, font_scale=args.font_scale)

        # out_file = os.path.join(args.out_dir, str(1)+'.jpg')
        # cv2.imwrite(out_file, img)
        vw.write(image)
        # print("Output is saved at {}".format(out_file))
        # if i > args.num_imgs:
        #     break
    vw.release()
    vid.release()
