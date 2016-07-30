# coding:utf-8
"""
Created on 16/6/25 21:48 2016

@author: harry sun
"""
import os
import numpy as np
import pandas as pd
import subprocess
import shlex
import scipy.io
from PIL import Image, ImageDraw


def get_correct_boxes_list(boxes_list):
    subtractor = np.array((1, 1, 0, 0))

    # todo switch this to use map/reduce
    correct_boxes_list = []
    for boxes in boxes_list:
        correct_boxes = boxes - subtractor
        correct_boxes[2:4] = correct_boxes[0:2] + correct_boxes[2:4]
        correct_boxes_list.append(correct_boxes)

    return correct_boxes_list


def draw_boxes(img, box, color):
    # print('Totally %d boxes' % (len(boxes_list)))

    dr = ImageDraw.Draw(img)
    box = map(int, tuple(box))
    dr.rectangle(box, outline=color)
    return img

def listdir_no_hidden(path):
    list1 = []
    for f in sorted(os.listdir(path)):
        if not f.startswith('.'):
            p = os.path.abspath(path)
            list1.append(os.path.join(p, f))
    return list1


def write_list_to_csv(list1, path_out, header=False):
    temp = pd.DataFrame(list1)
    temp.to_csv(path_out, index=False, header=header)


def save_gt_roidb_csv(data_path, csv_path, out_path):
    box_list = pd.read_csv(csv_path, header=0).get_values()

    image_path_list = listdir_no_hidden(data_path)
    assert len(box_list) == len(image_path_list), 'the length of box list must equal to image list'
    new_list = []
    new_list1 = []
    for idx, entry in enumerate(image_path_list):
        s1 = str(entry)
        temp = box_list[idx]
        # change the x y coordination to correct [X1 Y1 X2 Y2]
        x1 = str(temp[-2])
        y1 = str(temp[-4])
        x2 = str(temp[-1])
        y2 = str(temp[-3])
        s2 = x1+' '+ y1+' '+x2+' '+y2
        new_list.append(s1 + ' 1 ' + s2)
        new_list1.append(s1)
    write_list_to_csv(new_list, out_path)
    write_list_to_csv(new_list1, './data_file/image_index_list.csv')


if __name__ == '__main__':
    datasets_path = '../DatabaseEars/'
    csv_path = os.path.join(datasets_path, 'boundaries.csv')
    image_path = os.path.join(datasets_path, 'DatabaseEars/')
    output_path = os.path.join('./data_file/gt_roidb.csv')

    # image_path_list = ['../2.jpg']

    import matlab_wrapper
    matlab = matlab_wrapper.MatlabSession()
    matlab.eval("cd('/home/harrysocool/Github/fast-rcnn/OP_methods/edges')")
    matlab.eval("addpath(genpath('/home/harrysocool/Github/fast-rcnn/OP_methods/edges'))")
    # matlab.eval("toolboxCompile")

    print("success loaded")

    # save_gt_roidb_csv(image_path, csv_path, output_path)

    index = 145
    list1 = pd.read_csv(output_path, header=None).values.flatten().tolist()
    l = list1[index].split(' ')

    ALPHA = 0.20 #step size of sliding window search
    BETA = 0.05  #nms threshold for object proposals
    model_name = 'EAR0.4_2'

    matlab.eval("res = edge_detector_demo('"+l[0]+"' ,'"+model_name+"',"+str(ALPHA)+' ,'+str(BETA)+")")
    temp_all_boxes_list = matlab.get('res')
    all_boxes_list = temp_all_boxes_list.tolist()

    img = Image.open(l[0])
    correct_boxes_list = get_correct_boxes_list(all_boxes_list)
    for box in correct_boxes_list:
        img = draw_boxes(img, box, 'red')
    I = img

    I2 = draw_boxes(I, (l[-4], l[-3], l[-2], l[-1]), 'green')
    I2.show()
    # pass