# -*- coding: utf-8 -*-
"""
Created on 16/7/1 15:20 2016

@author: harry sun
"""
import shlex
import subprocess

./tools/train_net.py --solver models/VGG_CNN_M_1024/solver.prototxt --weights data/imagenet_models/VGG_CNN_M_1024.v2.caffemodel  --rand --imdb soton_ear
pid = subprocess.Popen(
    cmd.split())
retcode = pid.wait()
