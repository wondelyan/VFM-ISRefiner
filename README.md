# VFM-ISRefiner
Deliang Wang,
Peng Liu,
Rongkai Zhuang,
Lajiao Chen,
Bing Li,
and Yi Zeng

Code for **"VFM-ISRefiner: Towards Better Adapting Vision Foundation Models for Interactive Segmentation of Remote Sensing Images"**

### Model Structure

![image](https://github.com/wondelyan/VFM-ISRefiner/blob/main/image/VFM-ISRefiner_Structure.png)


### Requirements
- PyTorch 1.11.0
- CUDA 11.3
- CuDNN 8.2.0
- python 3.8
  
### Installation
Create conda environment:
```bash
    $ conda create -n IS_ENV python=3.8 anaconda
    $ conda activate IS_ENV
    $ conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch
    $ pip install -r requirements.txt
```
Download repository:
```bash
    $ git clone https://github.com/wondelyan/VFM-ISRefiner.git
```
Download weights:

VFM-ISRefiner model [Baidu Netdisk](https://pan.baidu.com/s/1VoSG63g0pNE7yUXroyO9Ww?pwd=q52j)

### Datasets
The datasets used for training are iSAID-train and Potsdam. [Baidu Netdisk](https://pan.baidu.com/s/15nMOMJKpNDoBek78oDXKGg?pwd=vg4t)

The datasets used for testing are iSAID-val, SandBar, NWPU, LoveDA Urban and WHUBuilding-test. [Baidu Netdisk](https://pan.baidu.com/s/1ew8b8sWbsn6tBQHyjvGfng?pwd=ui34)
 

### Evaluation
For evaluation, please download the datasets and models, and then configure the path in [config.yml](https://github.com/wondelyan/VFM-ISRefiner/blob/main/config.yml)

```
python scripts/evaluate_model.py NoBRS \
--gpu=0 \
--checkpoint=./weights/model_checkpoints/VFM-ISRefiner_ViT_base448_RSImage.pth \
--eval-mode=cvpr \
--datasets=iSAID,SandBar,NWPU,LoveDA,WHUBuilding
```
### Train
For training, please download the MFP pretrained weights (click to download: [MFP_vit_Base](https://drive.google.com/drive/folders/1ygeSwkVfGlydP-LW6YnhSCyed4kLOe0f)) and put the weights in "./weights/pretrained" folder. 

Configure the dowloaded path in [config.yml](https://github.com/wondelyan/VFM-ISRefiner/blob/main/config.yml).

```
python train.py models/iter_mask/plainvit_base448_RSImage_itermask_prevMod.py \
--batch-size=8 \
--ngpus=1 \
--gpus=0
```

### Acknowledgement
Our project is developed upon [SimpleClick](https://github.com/uncbiag/SimpleClick) and [MFP](https://github.com/cwlee00/MFP). We sincerely thank for their outstanding contributions.
