# CHABCNet

> [ABCNet](https://github.com/aim-uofa/AdelaiDet) on the Chinese dataset.


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
pip install opencv-python
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


### Train Your Own Models

1. Step one: [Data processing](https://github.com/Eurus-Holmes/CHABCNet/tree/master/data_processing)

2. Step two: Put the ReCTS dataset to `dataset` folder


```
ReCTS:
.
├── images
│   ├── 000001.jpg
│   ├── 000002.jpg
├── annotations
│   ├── train.json
```

3. Step three: specify train img and annotations in `adet/data/builtin.py`:

Add the ReCTS dataset:

```
_PREDEFINED_SPLITS_TEXT = {
    "totaltext_train": ("totaltext/train_images", "totaltext/train.json"),
    "totaltext_val": ("totaltext/test_images", "totaltext/test.json"),
    "ctw1500_word_train": ("CTW1500/ctwtrain_text_image", "CTW1500/annotations/train_ctw1500_maxlen100_v2.json"),
    "ctw1500_word_test": ("CTW1500/ctwtest_text_image","CTW1500/annotations/test_ctw1500_maxlen100.json"),
    "syntext1_train": ("syntext1/images", "syntext1/annotations/train.json"),
    "syntext2_train": ("syntext2/images", "syntext2/annotations/train.json"),
    "mltbezier_word_train": ("mlt2017/images","mlt2017/annotations/train.json"),
    "ReCTS_train": ("ReCTS/train_images","ReCTS/annotations/train.json"),
    "ReCTS_test": ("ReCTS/test_images","ReCTS/annotations/test.json"),
}
```

4. Step four: specify train config in `configs/BAText/TotalText/Base-TotalText.yaml`:

Add the ReCTS dataset:

```
DATASETS:
  TRAIN: ("ReCTS_train",)
  TEST: ("ReCTS_test",)
```

5. Step five: run 

`OMP_NUM_THREADS=1 python tools/train_net2.py --config-file configs/BAText/CTW1500/attn_R_50.yaml --num-gpus 1`

`OMP_NUM_THREADS=1 python tools/train_net.py --config-file configs/BAText/TotalText/attn_R_50.yaml --num-gpus 1`

`OMP_NUM_THREADS=1 python tools/train_net2.py --config-file configs/BAText/Pretrain/attn_R_50.yaml --num-gpus 1`


