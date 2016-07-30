# coding:utf-8
"""
Created on 16/6/25 21:48 2016

@author: harry sun
"""
import os
import numpy as np
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


if __name__ == '__main__':
    datasets_path = '../DatabaseEars/'
    csv_path = os.path.join(datasets_path, 'boundaries.csv')
    image_path = os.path.join(datasets_path, 'DatabaseEars/')
    output_path = os.path.join('./data_file/gt_roidb.csv')

    # image_path_list = ['../2.jpg']

    import matlab_wrapper
    matlab = matlab_wrapper.MatlabSession()

    print("success loaded")

    # save_gt_roidb_csv(image_path, csv_path, output_path)

    index = 145
    list1 = pd.read_csv(output_path, header=None).values.flatten().tolist()
    l = list1[index].split(' ')

    ALPHA = 0.20 #step size of sliding window search
    BETA = 0.05  #nms threshold for object proposals
    model_name = 'EAR0.4_2'

    # edge_detector OP_method
    # matlab.eval("cd('/home/harrysocool/Github/fast-rcnn/OP_methods/edges')")
    # matlab.eval("addpath(genpath('/home/harrysocool/Github/fast-rcnn/OP_methods/edges'))")
    # matlab.eval("toolboxCompile")
    # matlab.eval("res = edge_detector_demo('"+l[0]+"' ,'"+model_name+"',"+str(ALPHA)+' ,'+str(BETA)+")")

    # selective_search OP_method
    matlab.eval("cd('/home/harrysocool/Github/fast-rcnn/OP_methods/selective_search_ijcv_with_python')")
    matlab.eval("addpath(genpath('/home/harrysocool/Github/fast-rcnn/OP_methods/selective_search_ijcv_with_python'))")
    matlab.eval("res = selective_search_demo('" + l[0] + "')")

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