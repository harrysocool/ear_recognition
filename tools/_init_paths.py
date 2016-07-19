# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Set up paths for Fast R-CNN."""

import os.path as osp
import sys

def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(osp.abspath(__file__))
parent_dir = osp.split(this_dir)[0]
# print(parent_dir)

# Add caffe to PYTHONPATH
caffe_path = osp.join(parent_dir, 'caffe-fast-rcnn', 'python')
add_path('/Users/harrysocool/Github/caffe/python')

# Add lib to PYTHONPATH
lib_path = osp.join(parent_dir, 'lib')
add_path(lib_path)
