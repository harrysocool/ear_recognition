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


class OP_method(object):
    def __init__(self):
        self.selective_search = 'selective_search'
        self.edge_detector = 'edge_detector'
        #
        self.ss_boxes_outpath = os.path.abspath('./mat_file/ss_all_boxes.mat')
        self.ed_boxes_outpath = os.path.abspath('./mat_file/ed_all_boxes.mat')


def save_mat_boxes(image_fnames, output_filename, cmd):
    if cmd == 'selective_search':
        script_dirname = '../OP_methods/selective_search_ijcv_with_python/'
    elif cmd == 'edge_detector':
        script_dirname = '../OP_methods/edges/'

    # make the file path to be absolute path
    image_fnames = [os.path.abspath(entry) for entry in image_fnames]
    output_filename = os.path.abspath(output_filename)

    # make the file name list into the cammand format for shell execute
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

    # swap the columns of boxes
    # todo switch this to use map/reduce
    correct_boxes_list = []
    for boxes in all_boxes:
        correct_boxes = np.zeros((len(boxes), 4))
        correct_boxes[:, (0, 1, 2, 3)] = boxes[:, (1, 0, 3, 2)]
        correct_boxes_list.append(correct_boxes)

    return correct_boxes_list


def read_ed_mat_boxes(output_filename):
    # Read the results and undo Matlab's 1-based indexing.
    all_boxes = list(scipy.io.loadmat(output_filename)['all_boxes'][0])
    subtractor = np.array((1, 1, 0, 0))[np.newaxis, :]

    #todo switch this to use map/reduce
    correct_boxes_list = []
    for boxes in all_boxes:
        correct_boxes = np.zeros((len(boxes), 4))
        for idx in range(len(boxes)):
            correct_boxes[idx] = boxes[idx] - subtractor
            correct_boxes[idx][2:4] = correct_boxes[idx][0:2] + correct_boxes[idx][2:4]
        correct_boxes_list.append(correct_boxes)

    return correct_boxes_list


def draw_boxes(image_path, boxes_list):
    print('Totally %d boxes' % (len(boxes_list)))

    img = Image.open(image_path)
    dr = ImageDraw.Draw(img)
    for box in boxes_list:
        box = tuple(box)
        dr.rectangle(box, outline="red")
    img.show()


if __name__ == '__main__':
    method = OP_method()
    image_path_list = ['../3.jpg']

    save_mat_boxes(image_path_list, method.ss_boxes_outpath, cmd=method.selective_search)
    all_boxes_list = read_ss_mat_boxes(method.ss_boxes_outpath)
    draw_boxes(image_path_list[0], all_boxes_list[0][0:2])

    save_mat_boxes(image_path_list, method.ed_boxes_outpath, cmd=method.edge_detector)
    all_boxes_list = read_ed_mat_boxes(method.ed_boxes_outpath)
    draw_boxes(image_path_list[0], all_boxes_list[0][0:2])
