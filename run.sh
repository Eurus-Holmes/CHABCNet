#!/usr/bin/env bash
echo "Runtime Environment Variables:"
echo \$PATH=$PATH
echo \$LD_LIBRARY_PATH=$LD_LIBRARY_PATH

# source /miniconda/etc/profile.d/conda.sh
# conda activate python36

conda install -c conda-forge cudatoolkit-dev
pip install -U torch==1.5 torchvision==0.6 -f https://download.pytorch.org/whl/cu101/torch_stable.html
pip install pyyaml==5.1
pip install pycocotools==2.0.1
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
python setup.py build develop


pip install awscli

aws s3 --endpoint=https://blob.mr3.simcloud.apple.com cp s3://ABCNet9/data/datasets.tar.gz datasets
tar -xf datasets

pip install ruamel.yaml

python tools/train_net3.py --config-file configs/BAText/CTW1500/attn_R_50.yaml --num-gpus 8
# python tools/train_net3.py --config-file configs/BAText/TotalText/attn_R_50.yaml --num-gpus 8
# python whileTrue.py
