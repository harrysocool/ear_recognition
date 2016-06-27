# -*- coding: utf-8 -*-
"""
Created on 16/6/25 21:48 2016

@author: harry sun
"""
import os
import numpy as np
import subprocess
import shlex
import scipy.io
from PIL import Image, ImageDraw


def save_ss_mat_boxes(image_fnames, output_filename, cmd='selective_search'):
    script_dirname = '../OP_methods/selective_search_ijcv_with_python/'

    # make the file path to be absolute path
    image_fnames = [os.path.abspath(entry) for entry in image_fnames]
    output_filename = os.path.abspath(output_filename)
    # get
    fnames_cell = '{' + ','.join("'{}'".format(x) for x in image_fnames) + '}'
    command = "{}({}, '{}')".format(cmd, fnames_cell, output_filename)
    print(command)

    # Execute command in MATLAB.
    mc = "matlab -nojvm -r \"try; {}; catch; exit; end; exit\"".format(command)
    pid = subprocess.Popen(
        shlex.split(mc), stdout=open(output_filename, 'w'), cwd=script_dirname)
    retcode = pid.wait()
    if retcode != 0:
        raise Exception("Matlab script did not exit successfully!")

def read_ss_mat_boxes(output_filename):
    # Read the results and undo Matlab's 1-based indexing.
    all_boxes = list(scipy.io.loadmat(output_filename)['all_boxes'][0])
    subtractor = np.array((1, 1, 0, 0))[np.newaxis, :]
    all_boxes = [boxes - subtractor for boxes in all_boxes]

    return all_boxes

def hahaha():

    boxes_list = get_windows(image_path_list)[0]
    print('Totally %d boxes' % (len(boxes_list)))

    # swap the columns of boxes
    correct_boxes_list = np.zeros((len(boxes_list), 4))
    correct_boxes_list[:, (0,1,2,3)] = boxes_list[:, (1,0,3,2)]

    img = Image.open(image_path_list[0])
    dr = ImageDraw.Draw(img)
    for i in range(len(correct_boxes_list)):
        box = tuple(correct_boxes_list[i])
        dr.rectangle(box, outline="red")
    img.show()



if __name__ == '__main__':
    # image_path_list = ['../selective_search_ijcv_with_python/000015.jpg']
    image_path_list = ['../2.jpg']

    ss_boxes_output_path = './mat_file/ss_boxes.mat'
    save_ss_mat_boxes(image_path_list, ss_boxes_output_path)