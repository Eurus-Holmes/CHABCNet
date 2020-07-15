# Dataset: [ReCTS2019](https://rrc.cvc.uab.es/?ch=12&com=introduction)

[Tasks - ICDAR 2019 Robust Reading Challenge on Reading Chinese Text on Signboard](https://rrc.cvc.uab.es/?ch=12&com=tasks)

## Reference

  - [example scripts](https://universityofadelaide.box.com/s/fo7odnmqe370btm7sdotqve1c0zsu8p3)


The structure of data_processing folder as below.

```
.
├── README.md
├── change_json.py
├── classes.txt
├── dictionary_inv.json
├── images
│   ├── train_ReCTS_000001.jpg
│   ├── train_ReCTS_000002.jpg
│   ├── train_ReCTS_000003.jpg
│   ├── train_ReCTS_000004.jpg
│   └── train_ReCTS_000005.jpg
├── labels
│   ├── train_ReCTS_000001.json
│   ├── train_ReCTS_000002.json
│   ├── train_ReCTS_000003.json
│   ├── train_ReCTS_000004.json
│   └── train_ReCTS_000005.json
├── my_Bezier_generator2.py
├── my_generate_abcnet_json.py
└── pre_processing.py
```

## Step one: Preprocessing

```python
python pre_processing.py
```


## Step two: Processing the custom dataset with only four vertices

```python
python change_json.py
```


## Step three: Given polygonal annotation, generating bezier curve annotation

```pyhon
python my_Bezier_generator2.py
```


## Step four: Given bezier curve annotation, generating coco-like annotation format for training abcnet
    
```python
python my_generate_abcnet_json.py ./ train 0
```
    

## Step five: Rename
    
```python
python rename.py
```
