import os
import random
import numpy as np
import pandas as pd
import matlab_wrapper


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


def save_gt_roidb_csv(data_path, csv_path, image_index_output_path, gt_output_path, test_image_path, test_gt):
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
    # shuffle the idx of training set
    shuffle_idx = range(len(image_path_list))
    random.seed(641) # make it can be reproduce
    random.shuffle(shuffle_idx)
    train_idx = shuffle_idx[0:437]
    test_idx = shuffle_idx[437:]

    train_image_path = [new_list1[idx] for idx in train_idx]
    train_gt = [new_list[idx] for idx in train_idx]
    test_image_path_data = [new_list1[idx] for idx in test_idx]
    test_gt_data = [new_list[idx] for idx in test_idx]

    write_list_to_csv(train_gt, gt_output_path)
    write_list_to_csv(train_image_path, image_index_output_path)

    write_list_to_csv(test_gt_data, test_gt)
    write_list_to_csv(test_image_path_data, test_image_path)
def initialize_matlab():
    matlab = matlab_wrapper.MatlabSession()

    # edge_detector OP_method
    matlab.eval("cd('/home/harrysocool/Github/fast-rcnn/OP_methods/edges')")
    matlab.eval("addpath(genpath('/home/harrysocool/Github/fast-rcnn/OP_methods/edges'))")
    matlab.eval("toolboxCompile")

    # # selective_search OP_method
    # matlab.eval("cd('/home/harrysocool/Github/fast-rcnn/OP_methods/selective_search_ijcv_with_python')")
    # matlab.eval("addpath(genpath('/home/harrysocool/Github/fast-rcnn/OP_methods/selective_search_ijcv_with_python'))")

    return matlab

if __name__ == '__main__':
    datasets_path = '/home/harrysocool/Github/fast-rcnn/DatabaseEars/'
    csv_path = os.path.join(datasets_path, 'boundaries.csv')
    image_path = os.path.join(datasets_path, 'DatabaseEars/')
    gt_output_path = os.path.join(datasets_path, '../','ear_recognition/data_file/gt_roidb.csv')
    image_index_output_path = os.path.join(datasets_path, '../', 'ear_recognition/data_file/image_index_list.csv')
    mat_output_filename = os.path.join(datasets_path, '../','ear_recognition/data_file/all_boxes.mat')

    test_gt_output_path = os.path.join(datasets_path, '../','ear_recognition/data_file/test_gt_roidb.csv')
    test_image_index_output_path = os.path.join(datasets_path, '../', 'ear_recognition/data_file/test_image_index_list.csv')

    # save_gt_roidb_csv(image_path, csv_path, image_index_output_path, gt_output_path, test_image_index_output_path,
    #                   test_gt_output_path)

    matlab = initialize_matlab()

    list1 = pd.read_csv(image_index_output_path, header=None).values.flatten().tolist()
    cmd = 'edge_detector'

    # fnames_cell = "{" + ",".join("'{}'".format(x) for x in list1) + "}"
    fnames_cell = "{"
    for x in list1:
        fnames_cell += "'" + x+ "',"
    fnames_cell += "}"
    # fnames_cell = "{'"+list1[0]+"','"+list1[1] +"'}"
    command = "res = {}({}, '{}')".format(cmd, fnames_cell, mat_output_filename)
    print(command)

    matlab.eval(command)