#!/usr/bin/python
# -*- coding: utf-8 -*-
import json
import os
import sys
import cv2
import numpy as np
from shapely.geometry import *

filepath = './dictionary_inv.json'
cV2 = []
with open(filepath, 'r') as f:
    data = json.load(f)
    for key, value in data.items():
        # print(key, value)
        cV2.append(value)
# print(cV2)

if len(sys.argv) < 3:
    print("Usage: python convert_to_detectron_json.py root_path phase split")
    print("For example: python convert_to_detectron_json.py data train 100200")
    exit(1)
root_path = sys.argv[1]
phase = sys.argv[2]
split = int(sys.argv[3])
dataset = {
    'licenses': [],
    'info': {},
    'categories': [],
    'images': [],
    'annotations': []
}
with open(os.path.join(root_path, 'classes.txt')) as f:
    classes = f.read().strip().split()
for i, cls in enumerate(classes, 1):
    dataset['categories'].append({
        'id': i,
        'name': cls,
        'supercategory': 'beverage',
        'keypoints': ['mean',
                      'xmin',
                      'x2',
                      'x3',
                      'xmax',
                      'ymin',
                      'y2',
                      'y3',
                      'ymax',
                      'cross']  # only for BDN
    })


def get_category_id(cls):
    for category in dataset['categories']:
        if category['name'] == cls:
            return category['id']


_indexes = sorted([f.split('.')[0].split('_')[-1]
                   for f in os.listdir(os.path.join(root_path, 'abcnet_gen_labels'))])

if phase == 'train':
    indexes = [line for line in _indexes if int(
        line) >= split]  # only for this file
else:
    indexes = [line for line in _indexes if int(line) <= split]
j = 1
for index in indexes:
    # if int(index) >3: continue
    print('Processing: ' + index)
    im = cv2.imread(os.path.join(root_path, 'images/') + 'train_ReCTS_' + index + '.jpg')
    height, width, _ = im.shape
    dataset['images'].append({
        'coco_url': '',
        'date_captured': '',
        'file_name': index + '.jpg',
        'flickr_url': '',
        'id': int(index),
        'license': 0,
        'width': width,
        'height': height
    })
    anno_file = os.path.join(root_path, 'abcnet_gen_labels/') + 'train_ReCTS_' + index + '.txt'

    with open(anno_file) as f:
        lines = [line for line in f.readlines() if line.strip()]
        for i, line in enumerate(lines):
            pttt = line.strip().split('||||')
            parts = pttt[0].split(',')
            ct = pttt[-1].strip()

            cls = 'text'
            segs = [float(kkpart) for kkpart in parts[:16]]

            xt = [segs[ikpart] for ikpart in range(0, len(segs), 2)]
            yt = [segs[ikpart] for ikpart in range(1, len(segs), 2)]
            xmin = min([xt[0], xt[3], xt[4], xt[7]])
            ymin = min([yt[0], yt[3], yt[4], yt[7]])
            xmax = max([xt[0], xt[3], xt[4], xt[7]])
            ymax = max([yt[0], yt[3], yt[4], yt[7]])
            width = max(0, xmax - xmin + 1)
            height = max(0, ymax - ymin + 1)
            if width == 0 or height == 0:
                continue

            max_len = 100
            recs = [len(cV2) + 1 for ir in range(max_len)]

            ct = str(ct)
            print('rec', ct)

            for ix, ict in enumerate(ct):
                if ix >= max_len: continue
                if ict in cV2:
                    recs[ix] = cV2.index(ict)
                else:
                    recs[ix] = len(cV2)

            dataset['annotations'].append({
                'area': width * height,
                'bbox': [xmin, ymin, width, height],
                'category_id': get_category_id(cls),
                'id': j,
                'image_id': int(index),
                'iscrowd': 0,
                'bezier_pts': segs,
                'rec': recs
            })
            j += 1
folder = os.path.join(root_path, 'annotations')
if not os.path.exists(folder):
    os.makedirs(folder)
json_name = os.path.join(root_path, 'annotations/{}.json'.format(phase))
with open(json_name, 'w') as f:
    json.dump(dataset, f)
