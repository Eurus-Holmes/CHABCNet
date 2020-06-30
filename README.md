# AdelaiDet

> [Source](https://github.com/aim-uofa/AdelaiDet)


## Installation

```shell script
git clone https://github.com/aim-uofa/AdelaiDet.git
cd AdelaiDet
python -m venv env
source env/bin/activate
pip install --upgrade pip
``` 

First install Detectron2 following the official guide: [INSTALL.md](https://github.com/facebookresearch/detectron2/blob/master/INSTALL.md). Then build AdelaiDet with:
```shell script
pip install -U torch==1.5 torchvision==0.6 -f https://download.pytorch.org/whl/cu101/torch_stable.html 
pip install cython pyyaml==5.1
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install detectron2==0.1.3 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.5/index.html
python setup.py build develop
```

Some projects may require special setup, please follow their own `README.md` in [configs](configs).


## Quick Start 

### Inference with our trained Models

1. Select the model and config file above, for example, `configs/BAText/CTW1500/attn_R_50.yaml`.
2. Run the demo with

```
wget -O ctw1500_attn_R_50.pth https://universityofadelaide.box.com/shared/static/1bqpg9hijtn2rcooqjpffateguh9eeme.pth
python demo/demo.py \
    --config-file configs/BAText/CTW1500/attn_R_50.yaml \
    --input ./input/ \
    --output ./output/ \
    --opts MODEL.WEIGHTS ctw1500_attn_R_50.pth
```
or
```
wget -O tt_attn_R_50.pth https://cloudstor.aarnet.edu.au/plus/s/t2EFYGxNpKPUqhc/download
python demo/demo.py \
    --config-file configs/BAText/TotalText/attn_R_50.yaml \
    --input ./input/ \
    --output ./output/ \
    --opts MODEL.WEIGHTS tt_attn_R_50.pth
```
