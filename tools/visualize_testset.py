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

#------------------------------------------------------------------------------
#  Utilization
#------------------------------------------------------------------------------

def inference_detector(model, data):
    with torch.no_grad():
        result = model(return_loss=False, rescale=True, **data)
    return result


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
    for i, data in tqdm.tqdm(enumerate(dataset)):
        data = scatter(collate([data], samples_per_gpu=1), [args.device])[0]
        result = inference_detector(model, data)

        # test
        image = mmcv.imread(data['img_metas'][0][0]['filename'])    
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
        image = cvut.draw_bboxes(image, bboxes[:,:4], labels=labels, classnames=dataset.CLASSES, thickness=2, font_size=2, font_thickness=2)        

        # img = model.show_result(data['img_metas'][0][0]['filename'], result, score_thr=args.det_thr, show=False,  thickness=args.thickness, font_scale=args.font_scale)

        out_file = os.path.join(args.out_dir, os.path.basename(data['img_metas'][0][0]['ori_filename']))
        # cv2.imwrite(out_file, img)
        cv2.imwrite(out_file, image)
        print("Output is saved at {}".format(out_file))
        if i > 50:
            break
