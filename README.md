# Vehicles Detection Tracking Speed Estimation using pytorch and MMDetection

This is the thesis conducted while we are studying in Ho Chi Minh City University of Technology, Vietnam. In this project, we apply Deep Learning using Pytorch framework and based on MMDetection to do vehicles detection, tracking and speed estimation. The dataset is collected at the overpass in Ho Chi Minh City, Vietnam and labels by our team. You can find more information of our work in [Project summary](https://drive.google.com/file/d/1Ke5uFPAcKx4uvgqOWkhXF8xmB1Yqza_d/view?usp=sharing). 

**Our main work is summarized as following**
- We divided the work into four parts for development: Detection part, tracking part, speed estimation part and dataset, in which we only focus on reading papers, perceive those ideas and apply them to improve the results.
- For object detection, we only research and apply various network architecture such as RetinaNet, Faster R-CNN as well as recent techniques for object detection including ATSS, data Augmentation, Focal KL Loss, etc. to push the accuracy.
- For tracking and speed estimation, we focus on applying IOU tracker and modify it for stable tracking results; applying formular V=S/t for speed estimation. We mainly evaluate the tracking result by human visualization because of the limitation of label for those parts.
- Make new dataset: The main problem we encounter is GPU resources for train Deep Learning Network. If we utilized the existed dataset which is extremely large and heavy, we could not do on that. Hence, we need a new dataset which is liter and apply transfer learning technique to reach our target. The details of our dataset is in the later section.

**Structure of this README**
- Installation
- Dataset preparation
- Train
- Test
- Result
- Team's information
- Citation

## Installation
#### Create conda env
```bash
conda create -n vdts python=3.7 -y
conda activate vdts
```
#### Install Pytorch, Torchvision and mmdetection
```bash 
conda install pytorch=1.5 torchvision -c pytorch
pip install "git+https://github.com/open-mmlab/cocoapi.git#subdirectory=pycocotools"
pip install git+https://github.com/open-mmlab/mmdetection.git@v2.2.0
pip install mmcv==0.6.2
```
**Note**:  Make sure that your compilation CUDA version and runtime CUDA version match. You can check the supported CUDA version for precompiled packages on the [Pytorch website](https://pytorch.org/)

#### Install other requirements
- OS: Ubuntu 18.04
- Python: 3.7.9

```bash
pip install git+https://github.com/thuyngch/cvut
pip install future tensorboard
```

<!-- #### Install ttdet as package
```bash
pip install -e ./
``` -->


## Dataset preparation
- Download dataset from Google Drive [Link](https://drive.google.com/file/d/1EcfzRi7bHdZDAwIDBdBtyAIEsWmytqUa/view?usp=sharing) and unzip it.
- The data after extracted should following the following structure: <br>
![alt text](./readme_images/data_dir_format.png) <br>
- Make symblic link to the dataset you just downloaded from project directory:
```bash
ln -s <PATH TO DATASET> data
```
**For Example**, my dataset named `data` is located at `/home/tuan/Desktop`, I do the following command: <br> <br>
![alt text](./readme_images/ln_s.png)


The result in the image above is that I make the symblic link name `data` to the folder containing dataset.
<br>
## Train
- Run the following command in bash shell:
```bash
#!/usr/bin/env bash
set -e
CFG="atss_r50_fpn_1x_street"                                        # file name of config file
WORKDIR="../TS/checkpoints/transfer_weight/${CFG}"                  # directory for saving checkpoints during training
CONFIG="configs/street/${CFG}.py"                                   # path to your config file
GPUS=2                                                              # number of GPU while training
LOAD_FROM="../TS/checkpoints/pretrained/atss_r50_fpn_1x_coco.pth"   # Pretrain weight from COCO dataset
export CUDA_VISIBLE_DEVICES=0,1
bash tools/dist_train.sh $CONFIG $GPUS --work-dir $WORKDIR  --options DATA_ROOT=$DATA_ROOT --load_from $LOAD_FROM
```
- In the above example, config file is `configs/street/atss_r50_fpn_1x_street.py`, pretrained weight is `atss_r50_fpn_1x_coco.pth` and saved at `../TS/checkpoints/pretrained`. Checkpoints will save under `../TS/checkpoints/transfer_weight/atss_r50_fpn_1x_street`.

- NOTE: The pretrained weight from COCO is download at [MMDetection repo](https://github.com/open-mmlab/mmdetection), following section will give the specific link. 


## Test 
- Run the following command in bash shell:

```bash
#!/usr/bin/env bash
set -e
export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

CONFIG_FILE="atss_r18_fpn_2x_street"                            # file name of config file
WORKDIR="../TS/checkpoints/transfer_weight/${CONFIG_FILE}"      # directory of checkpoints
CONFIG="configs/street/${CONFIG_FILE}.py"                       # path to your config file
CHECKPOINT="${WORKDIR}/epoch_12.pth"                            # checkpoint file, in this case `epoch_12.pth`
RESULT="${WORKDIR}/epoch_12.pkl"

GPUS=2
export CUDA_VISIBLE_DEVICES=0,1

python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$((RANDOM + 10000)) \
    tools/test.py $CONFIG $CHECKPOINT --launcher pytorch --out $RESULT --eval bbox
```

- Run the following command to test the speed of network:

```bash
python tools/benchmark.py <YOUR_CONFIG_FILE.py>
```

## Result
Our results of the object detection method are summarized in the following table:
<!-- <style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
.tg .tg-kfkg{background-color:#dae8fc;border-color:inherit;text-align:center;vertical-align:top}
.tg .tg-n533{background-color:#dae8fc;border-color:inherit;font-weight:bold;text-align:center;vertical-align:top}
</style> -->
<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow"><span style="font-weight:bold">No.</span></th>
    <th class="tg-c3ow"><span style="font-weight:bold">Method</span></th>
    <th class="tg-c3ow"><span style="font-weight:bold">Albu</span></th>
    <th class="tg-c3ow"><span style="font-weight:bold">Multiscale training</span></th>
    <th class="tg-c3ow"><span style="font-weight:bold">Scheduler</span></th>
    <th class="tg-c3ow"><span style="font-weight:bold">mAP</span></th>
    <th class="tg-c3ow"><span style="font-weight:bold">FPS</span></th>
    <th class="tg-c3ow">Image size</th>
    <th class="tg-c3ow">Config</th>
    <th class="tg-c3ow">COCO Pretrained</th>
    <th class="tg-c3ow">Our weight</th>
    <th class="tg-c3ow">Note</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-kfkg">1</td>
    <td class="tg-kfkg">YOLOv3 (3fpn levels)</td>
    <td class="tg-kfkg">+Cutmix</td>
    <td class="tg-kfkg">yes</td>
    <td class="tg-kfkg">----</td>
    <td class="tg-kfkg">0.63</td>
    <td class="tg-kfkg"></td>
    <td class="tg-kfkg">---</td>
    <td class="tg-kfkg">x</td>
    <td class="tg-kfkg"></td>
    <td class="tg-kfkg"></td>
    <td class="tg-kfkg" rowspan="3"><br><br>Original</td>
  </tr>
  <tr>
    <td class="tg-kfkg">2</td>
    <td class="tg-kfkg">Vanila RetinaNet - R50</td>
    <td class="tg-kfkg">no</td>
    <td class="tg-kfkg">no</td>
    <td class="tg-kfkg">1x</td>
    <td class="tg-kfkg">0.632</td>
    <td class="tg-kfkg">18.2</td>
    <td class="tg-kfkg">1518x800</td>
    <td class="tg-kfkg"><a href="https://github.com/TuanTNG/Vehicles-Detection-Tracking-Speed-estimation-pytorch-mmdet/blob/master/configs/street/retinanet_r50_fpn_1x_street.py" target="_blank" rel="noopener noreferrer">retinanet_r50_fpn_1x_street.py</a></td>
    <td class="tg-kfkg"><a href="http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_2x_coco/retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth" target="_blank" rel="noopener noreferrer">retinanet_r50_fpn_2x_coco.pth</a></td>
    <td class="tg-kfkg"><a href="https://1drv.ms/u/s!AmoaChPnSuIOg3QZgpqDXhOSi3k8?e=2puiPq" target="_blank" rel="noopener noreferrer">retinanet_r50_fpn_1x_street</a></td>
  </tr>
  <tr>
    <td class="tg-kfkg">3</td>
    <td class="tg-kfkg">Faster R-CNN</td>
    <td class="tg-kfkg">no</td>
    <td class="tg-kfkg">no</td>
    <td class="tg-kfkg">1x</td>
    <td class="tg-kfkg">0.481</td>
    <td class="tg-kfkg">x</td>
    <td class="tg-kfkg">x</td>
    <td class="tg-kfkg"><a href="https://github.com/TuanTNG/Vehicles-Detection-Tracking-Speed-estimation-pytorch-mmdet/blob/master/configs/street/faster_rcnn_r50_fpn_1x_street.py" target="_blank" rel="noopener noreferrer">faster_rcnn_r50_fpn_1x_street.py</a></td>
    <td class="tg-kfkg"><a href="http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_2x_coco/faster_rcnn_r50_fpn_2x_coco_bbox_mAP-0.384_20200504_210434-a5d8aa15.pth" target="_blank" rel="noopener noreferrer">faster_rcnn_r50_fpn_2x_coco.pth</a></td>
    <td class="tg-kfkg"><a href="https://1drv.ms/u/s!AmoaChPnSuIOg25Xi9auHfDeSzrj?e=xgU3nm" target="_blank" rel="noopener noreferrer">faster_rcnn_r50_fpn_1x_coco</a></td>
  </tr>
  <tr>
    <td class="tg-c3ow">4</td>
    <td class="tg-c3ow">ATSS-R50</td>
    <td class="tg-c3ow">no</td>
    <td class="tg-c3ow">no</td>
    <td class="tg-c3ow">1x</td>
    <td class="tg-c3ow"><span style="font-weight:bold">0.747</span></td>
    <td class="tg-c3ow">17.9</td>
    <td class="tg-c3ow">---</td>
    <td class="tg-c3ow"><a href="https://github.com/TuanTNG/Vehicles-Detection-Tracking-Speed-estimation-pytorch-mmdet/blob/master/configs/street/atss_r50_fpn_1x_street.py" target="_blank" rel="noopener noreferrer">atss_r50_fpn_1x_street.py</a></td>
    <td class="tg-c3ow"><a href="http://download.openmmlab.com/mmdetection/v2.0/atss/atss_r50_fpn_1x_coco/atss_r50_fpn_1x_coco_20200209-985f7bd0.pth" target="_blank" rel="noopener noreferrer">atss_r50_fpn_1x_coco.pth</a></td>
    <td class="tg-c3ow"><a href="https://1drv.ms/u/s!AmoaChPnSuIOg2aXjOGjIdC41Oca?e=GVPQiq" target="_blank" rel="noopener noreferrer">atss_r50_fpn_1x_street</a></td>
    <td class="tg-c3ow">Baseline</td>
  </tr>
  <tr>
    <td class="tg-kfkg">5</td>
    <td class="tg-kfkg">ATSS+Net-R18</td>
    <td class="tg-kfkg">no</td>
    <td class="tg-kfkg">no</td>
    <td class="tg-kfkg">1x</td>
    <td class="tg-kfkg">0.522</td>
    <td class="tg-kfkg">30</td>
    <td class="tg-kfkg">---</td>
    <td class="tg-kfkg"></td>
    <td class="tg-kfkg"></td>
    <td class="tg-kfkg"></td>
    <td class="tg-kfkg" rowspan="3"><br><br>Different <br><br>Backbones</td>
  </tr>
  <tr>
    <td class="tg-kfkg">6</td>
    <td class="tg-kfkg">ATSS+MobileNetV3</td>
    <td class="tg-kfkg">no</td>
    <td class="tg-kfkg">no</td>
    <td class="tg-kfkg">1x</td>
    <td class="tg-kfkg">0.646</td>
    <td class="tg-kfkg">32.5</td>
    <td class="tg-kfkg">---</td>
    <td class="tg-kfkg"></td>
    <td class="tg-kfkg"></td>
    <td class="tg-kfkg"></td>
  </tr>
  <tr>
    <td class="tg-kfkg">7</td>
    <td class="tg-kfkg">Vanila RetinaNet with MobileNetv3</td>
    <td class="tg-kfkg">no</td>
    <td class="tg-kfkg">no</td>
    <td class="tg-kfkg">1x</td>
    <td class="tg-kfkg">0.38</td>
    <td class="tg-kfkg">x</td>
    <td class="tg-kfkg">---</td>
    <td class="tg-kfkg"></td>
    <td class="tg-kfkg"></td>
    <td class="tg-kfkg"></td>
  </tr>
  <tr>
    <td class="tg-c3ow">8</td>
    <td class="tg-c3ow">ATSS-R50</td>
    <td class="tg-c3ow">yes</td>
    <td class="tg-c3ow">no</td>
    <td class="tg-c3ow">1x</td>
    <td class="tg-c3ow">0.686</td>
    <td class="tg-c3ow">---</td>
    <td class="tg-c3ow">---</td>
    <td class="tg-c3ow"><a href="https://github.com/TuanTNG/Vehicles-Detection-Tracking-Speed-estimation-pytorch-mmdet/blob/master/configs/street/atss_r50_fpn_albu_1x_street.py" target="_blank" rel="noopener noreferrer">atss_r50_fpn_albu_1x_street.py</a></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow">Augment</td>
  </tr>
  <tr>
    <td class="tg-kfkg">9</td>
    <td class="tg-kfkg">ATSS-R50</td>
    <td class="tg-kfkg">no</td>
    <td class="tg-kfkg">no</td>
    <td class="tg-kfkg">1x</td>
    <td class="tg-n533">0.759</td>
    <td class="tg-kfkg">---</td>
    <td class="tg-kfkg">4096x3072</td>
    <td class="tg-kfkg"></td>
    <td class="tg-kfkg"></td>
    <td class="tg-kfkg"></td>
    <td class="tg-kfkg">Big size</td>
  </tr>
  <tr>
    <td class="tg-c3ow">10</td>
    <td class="tg-c3ow">ATSS-R50</td>
    <td class="tg-c3ow">no</td>
    <td class="tg-c3ow">no</td>
    <td class="tg-c3ow">1x</td>
    <td class="tg-c3ow">0.679</td>
    <td class="tg-c3ow">---</td>
    <td class="tg-c3ow">---</td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow" rowspan="3"><br><br>lr=1e-3</td>
  </tr>
  <tr>
    <td class="tg-c3ow">11</td>
    <td class="tg-c3ow">ATSS-R50 (3fpn levels)</td>
    <td class="tg-c3ow">no</td>
    <td class="tg-c3ow">no</td>
    <td class="tg-c3ow">1x</td>
    <td class="tg-c3ow">0.656</td>
    <td class="tg-c3ow">---</td>
    <td class="tg-c3ow">---</td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
  </tr>
  <tr>
    <td class="tg-c3ow">12</td>
    <td class="tg-c3ow">ATSS-R50</td>
    <td class="tg-c3ow">yes</td>
    <td class="tg-c3ow">no</td>
    <td class="tg-c3ow">1x</td>
    <td class="tg-c3ow">0.667</td>
    <td class="tg-c3ow">---</td>
    <td class="tg-c3ow">---</td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
  </tr>
  <tr>
    <td class="tg-kfkg">13</td>
    <td class="tg-kfkg">ATSS-R50</td>
    <td class="tg-kfkg">no</td>
    <td class="tg-kfkg">yes</td>
    <td class="tg-kfkg">2x</td>
    <td class="tg-kfkg">0.728</td>
    <td class="tg-kfkg">---</td>
    <td class="tg-kfkg">---</td>
    <td class="tg-kfkg"><a href="https://github.com/TuanTNG/Vehicles-Detection-Tracking-Speed-estimation-pytorch-mmdet/blob/master/configs/street/atss_r50_fpn_2x_street.py" target="_blank" rel="noopener noreferrer">atss_r50_fpn_2x_street.py</a></td>
    <td class="tg-kfkg"></td>
    <td class="tg-kfkg"></td>
    <td class="tg-kfkg" rowspan="3"><br><br><br>2x</td>
  </tr>
  <tr>
    <td class="tg-kfkg">14</td>
    <td class="tg-kfkg">ATSS-R50</td>
    <td class="tg-kfkg">yes</td>
    <td class="tg-kfkg">yes</td>
    <td class="tg-kfkg">2x</td>
    <td class="tg-n533">0.75</td>
    <td class="tg-kfkg">---</td>
    <td class="tg-kfkg">---</td>
    <td class="tg-kfkg"></td>
    <td class="tg-kfkg"></td>
    <td class="tg-kfkg"></td>
  </tr>
  <tr>
    <td class="tg-kfkg">15</td>
    <td class="tg-kfkg">ATSS-R50</td>
    <td class="tg-kfkg">no</td>
    <td class="tg-kfkg">no</td>
    <td class="tg-kfkg">2x</td>
    <td class="tg-kfkg">0.747</td>
    <td class="tg-kfkg">---</td>
    <td class="tg-kfkg">---</td>
    <td class="tg-kfkg"></td>
    <td class="tg-kfkg"></td>
    <td class="tg-kfkg"></td>
  </tr>
  <tr>
    <td class="tg-c3ow">16</td>
    <td class="tg-c3ow">ATSS-R50 - Focal KL loss</td>
    <td class="tg-c3ow">no</td>
    <td class="tg-c3ow">no</td>
    <td class="tg-c3ow">1x</td>
    <td class="tg-c3ow">0.607</td>
    <td class="tg-c3ow">---</td>
    <td class="tg-c3ow">---</td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow"></td>
    <td class="tg-c3ow">Focal KL Loss</td>
  </tr>
</tbody>
</table>

**NOTE** COCO trained weights are taken from [MMDetection repo](https://github.com/open-mmlab/mmdetection).

## Team member
- [Tuan Tang Ngoc](https://tuantng.github.io/) - Develop object detection part
- Nam Cao Van - Develop object tracking and speed estimation part
- [Ph.D. Hao Nguyen Vinh](https://www.researchgate.net/profile/Vinh_Hao_Nguyen) - Supervisor

## Citation
```
@article{vdtse,
  title   = {Vehicles Detection Tracking Speed Estimation},
  author  = {Tuan Tang Ngoc, Nam Cao Van, Hao Nguyen Vinh},
  journal= {https://github.com/TuanTNG/Vehicles-Detection-Tracking-Speed-estimation-pytorch-mmdet},
  year={2020}
}
```