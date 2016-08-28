import csv
import os
import random
import numpy as np
import pandas as pd
import matlab_wrapper
from lib.utils.timer import Timer
from tools.ear_recog import get_gt, ROI_boxes


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

def time_analyse(matlab, cmd, image_filepath, par1, par2):
    timer = Timer()
    timer.tic()

    obj_proposals = ROI_boxes(matlab, image_filepath, cmd, par1, par2)

    timer.toc()
    time = timer.total_time
    box_numer = len(obj_proposals)

    return time, box_numer, obj_proposals

def mean_IOU_ratio(image_index, dets):
    ratio = np.empty(0,dtype=np.float64)
    (x1, y1, x2, y2) = get_gt(image_index)
    for box in dets:
        X1 = box[0]
        Y1 = box[1]
        X2 = box[2]
        Y2 = box[3]
        if ((np.float32(x1)-X1)<=15 and (X2- np.float32(x2))<=15
            and (np.float32(y1)-Y1)<=15 and (Y2-np.float32(y2))<=15):
            ratio = np.append(ratio,1.0)
        else:
            SI = max(0, min(x2, X2) - max(x1, X1)) * \
                 max(0, min(y2, Y2) - max(y1, Y1))
            SU = (x2 - x1) * (y2 - y1) + (X2 - X1) * (Y2 - Y1) - SI
            ratio = np.append(ratio, SI/SU)
    if ratio.size == 0:
        big_ratio = 0
    else:
        big = np.where(ratio >= 0.1)[0].size
        total = float(len(dets))
        big_ratio = float(big/total)
    return big_ratio

if __name__ == '__main__':
    datasets_path = '/home/harrysocool/Github/fast-rcnn/DatabaseEars'
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
    timer = Timer()

    list1 = pd.read_csv(test_image_index_output_path, header=None).values.flatten().tolist()
    cmd = 'ss'
    # ks = [50 100 150 200 300];
    par2_list = [2,3,4]
    # par2_list = [3]
    time_csv_out_path = os.path.join(os.path.dirname(datasets_path), 'result', cmd + '_' + 'OPtune_result_1.csv')

    # list2 = []
    # for par2 in par2_list:
    #     for par1 in [1]:
    #         for index, image_path in enumerate(list1):
    #             if index>300:
    #                 break
    #             time, box_numer, obj_proposals = time_analyse(matlab, cmd, image_path, par1, par2)
    #             ratio = mean_IOU_ratio(index + 1, obj_proposals)
    #             # list2.append([time, box_numer])
    #             # print('{} has processed in {:.3f} seconds with {} boxes'.format(len(list2), time, box_numer))
    #             print('No. {} has processed with par {} {}, box {} IOU ratio {:.3f} in {:.2f} seconds'.format(index,
    #                                                                                                       par1, par2,box_numer ,ratio, time))
    #             with open(time_csv_out_path, 'a') as csvfile:
    #                 writer = csv.writer(csvfile)
    #                 writer.writerow([par1, par2,ratio,box_numer, time])

    # write_list_to_csv(list2, time_csv_out_path)

    fnames_cell = "{" + ",".join("'{}'".format(x) for x in list1) + "}"
    command = "res = {}({}, '{}')".format('selective_search', fnames_cell, mat_output_filename)
    print(command)
    #
    # matlab.eval(command)