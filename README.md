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
- Train the network
- Inference
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
```bash
pip install git+https://github.com/thuyngch/cvut
pip install future tensorboard
```

#### Install ttdet as package
```bash
pip install -e ./
```

#### Make soft directory to data folder
```bash
ln -s /home/cybercore/tank/TS/data/thesisdata data
```

## Dataset preparation
- Download dataset from Google Drive [Link]()