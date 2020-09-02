import os, cv2
import numpy as np
from glob import glob
import torch, mmcv, argparse

from mmcv.runner import load_checkpoint
from mmcv.parallel import collate, scatter

from mmdet.apis.inference import LoadImage
from mmdet.datasets.pipelines import Compose

from ccdet.apis import init_detector
from ccdet.models import build_detector

import numpy as np
import matplotlib.pyplot as plt
import os, cv2, mmcv, torch, cvut
from ccdet.datasets import build_dataset
import os

#------------------------------------------------------------------------------
#  Utilization
#------------------------------------------------------------------------------
def get_data(img, cfg, device):
    # import ipdb; ipdb.set_trace()
    test_pipeline = Compose([LoadImage()] + cfg.test_pipeline[1:])
    data = dict(img=img)
    data = test_pipeline(data)
    data = scatter(collate([data], samples_per_gpu=1), [device])[0]
    return data

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


args = parser.parse_args()


#------------------------------------------------------------------------------
#  Main
#------------------------------------------------------------------------------
if __name__ == "__main__":
    # Data
    _SUPPORT_IMG_FORMAT = ['jpg', 'jpeg', 'png']
    img_files = sorted(glob(os.path.join(args.data_dir, "*.*")))
    img_files = [
        img_file for img_file in img_files
        if img_file.split('.')[-1].lower() in _SUPPORT_IMG_FORMAT]

    img_files = img_files[:args.num_imgs]
    print("Number of files {}".format(len(img_files)))

    # Build model
    model = init_detector(args.cfg, args.ckpt, device='cuda')
    os.makedirs(args.out_dir, exist_ok=True)

    # Inference
    for img_file in img_files:
        data = get_data(img_file, model.cfg, next(model.parameters()).device)
        result = inference_detector(model, data)
        img = model.show_result(img_file, result, score_thr=args.det_thr, show=False,  thickness=args.thickness, font_scale=args.font_scale)

        out_file = os.path.join(args.out_dir, os.path.basename(img_file))
        cv2.imwrite(out_file, img)
        print("Output is saved at {}".format(out_file))
