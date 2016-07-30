import os
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

def initialize_matlab():
    matlab = matlab_wrapper.MatlabSession()

    # edge_detector OP_method
    # matlab.eval("cd('/home/harrysocool/Github/fast-rcnn/OP_methods/edges')")
    # matlab.eval("addpath(genpath('/home/harrysocool/Github/fast-rcnn/OP_methods/edges'))")
    # matlab.eval("toolboxCompile")

    # selective_search OP_method
    matlab.eval("cd('/home/harrysocool/Github/fast-rcnn/OP_methods/selective_search_ijcv_with_python')")
    matlab.eval("addpath(genpath('/home/harrysocool/Github/fast-rcnn/OP_methods/selective_search_ijcv_with_python'))")

    return matlab

if __name__ == '__main__':
    datasets_path = '/home/harrysocool/Github/fast-rcnn/DatabaseEars/'
    csv_path = os.path.join(datasets_path, 'boundaries.csv')
    image_path = os.path.join(datasets_path, 'DatabaseEars/')
    csv_output_path = os.path.join(datasets_path, '../','ear_recognition/data_file/image_index_list.csv')
    mat_output_filename = os.path.join(datasets_path, '../','ear_recognition/data_file/all_boxes.mat')

    # save_gt_roidb_csv(image_path, csv_path, csv_output_path)

    matlab = initialize_matlab()

    list1 = pd.read_csv(csv_output_path, header=None).values.flatten().tolist()

    cmd = 'selective_search'

    fnames_cell = '{' + ','.join("'{}'".format(x) for x in list1) + '}'
    command = "res = {}({}, '{}')".format(cmd, fnames_cell, mat_output_filename)

    matlab.eval(command)