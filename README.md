# Vehicles-Detection-Tracking-Speed-estimation-pytorch-mmdet

## Installation
### Create conda env
```bash
conda create -n vdts python=3.7 -y
conda activate vdts
```
### Install Pytorch, Torchvision and mmdetection
```bash 
conda install pytorch=1.5 torchvision -c pytorch
pip install "git+https://github.com/open-mmlab/cocoapi.git#subdirectory=pycocotools"
pip install git+https://github.com/open-mmlab/mmdetection.git@v2.2.0
pip install mmcv==0.6.2
```
**Note**:  Make sure that your compilation CUDA version and runtime CUDA version match. You can check the supported CUDA version for precompiled packages on the [Pytorch website](https://pytorch.org/)

### Install other requirements
```bash
pip install git+https://github.com/thuyngch/cvut
pip install future tensorboard
```

### Install ttdet as package
```bash
pip install -e ./
```

### Make soft directory to data folder
```bash
ln -s /home/cybercore/tank/TS/data/thesisdata data
```
