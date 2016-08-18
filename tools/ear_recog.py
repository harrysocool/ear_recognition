import csv
import linecache
import _init_paths
from lib.fast_rcnn.config import cfg
from lib.fast_rcnn.test import im_detect
from lib.utils.nms import nms
from lib.utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
from OP_methods.BING_Objectness.source.bing_demo import bing_demo

count = 0
false_count = 0
false_positive_count = 0
CLASSES = ('__background__', 'ear')

NETS = {'vgg16': ('VGG16',
                  'vgg16_fast_rcnn_iter_40000.caffemodel'),
        'vgg_cnn_m_1024': ('VGG_CNN_M_1024',
                           'vgg_cnn_m_1024_fast_rcnn_iter_40000.caffemodel'),
        'caffenet': ('CaffeNet',
                     'caffenet_fast_rcnn_iter_40000.caffemodel')}

OP_method = ('ss', 'ed', 'BING')


def IOU_ratio(image_index, dets):
    ratio = []
    (x1, y1, x2, y2) = get_gt(image_index)
    for box in dets:
        X1 = box[0]
        Y1 = box[1]
        X2 = box[2]
        Y2 = box[3]
        SI = max(0, min(x2, X2) - max(x1, X1)) *\
             max(0, min(y2, Y2) - max(y1, Y1))
        SU = (x2-x1)*(y2-y1) + (X2-X1)*(Y2-Y1)  - SI
        ratio.append(SI/SU)
    return ratio


def get_gt(image_index):
    gt_csv_path = os.path.join(cfg.ROOT_DIR, 'ear_recognition', 'data_file', 'test_gt_roidb.csv')
    line = linecache.getline(gt_csv_path, image_index).strip('\n').split()
    x1 = int(line[-4])
    y1 = int(line[-3])
    x2 = int(line[-2])
    y2 = int(line[-1])
    return (x1, y1, x2, y2)


def ROI_boxes(matlab, image_filepath, cmd):
    if cmd == 'ed':
        # add the matlab directory path
        matlab.eval("cd('/home/harrysocool/Github/fast-rcnn/OP_methods/edges')")
        matlab.eval("addpath(genpath('/home/harrysocool/Github/fast-rcnn/OP_methods/edges'))")
        matlab.eval("toolboxCompile")
        matlab.eval("res = edge_detector_demo('{}','{}',{},{})".format(image_filepath, 'EAR0.4_2', 0.55, 0.75))
        raw_boxes = matlab.get('res')
        boxes = np.asarray(raw_boxes)
    elif cmd == 'ss':
        # selective_search OP_method
        matlab.eval("cd('/home/harrysocool/Github/fast-rcnn/OP_methods/selective_search_ijcv_with_python')")
        matlab.eval(
            "addpath(genpath('/home/harrysocool/Github/fast-rcnn/OP_methods/selective_search_ijcv_with_python'))")
        matlab.eval("res = selective_search_demo('{}')".format(image_filepath))
        raw_boxes = matlab.get('res')
        boxes = np.asarray(raw_boxes)
    elif cmd == 'BING':
        # BING method
        boxes, _ = bing_demo(image_filepath)
    else:
        raise NameError('Wrong ROI OP_methods name. (CHOOSE FROM: ss, ed, BING)')

    return boxes


def transform_image(image_index, cmd, variable=None):
    assert variable != None, 'No variable input, needs more variable input'
    index_csv_path = os.path.join(cfg.ROOT_DIR, 'ear_recognition', 'data_file', 'test_image_index_list.csv')
    image_filepath = linecache.getline(index_csv_path, image_index).strip('\n')

    # craete the folder for transform iamge
    datasets_path = '/home/harrysocool/Github/fast-rcnn/DatabaseEars/'
    file_name = os.path.basename(image_filepath)
    dir_path = os.path.join(datasets_path, cmd + '_' + str(variable))
    new_image_filepath = os.path.join(dir_path, file_name)
    # check for the exsitence of folders
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    # check for the exsitence of files
    elif os.path.exists(new_image_filepath):
        return new_image_filepath

    im = cv2.imread(image_filepath)
    if cmd == 'noise':
        float_im = np.float64(im)
        noise = np.random.randn(im.shape) * variable
        noise_im = float_im + noise
        noisy = np.uint8(np.clip(noise_im, 0, 255))
        cv2.imwrite(new_image_filepath, noisy)
    elif cmd == 'occlude':
        (x1, y1, x2, y2) = get_gt(image_index)
        new_y2 = round(y1 + (y2 - y1) * variable)
        mask = np.ones(im.shape[:2], np.uint8)
        mask[y1:new_y2, x1:x2] = 0
        new_im = cv2.bitwise_and(im, im, mask=mask)
        # cv2.imshow('sss', new_im)
        cv2.imwrite(new_image_filepath, new_im)
    return new_image_filepath


def demo(net, matlab, image_filepath, classes, method):
    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    # Load pre-computed Selected Search object proposals
    obj_proposals = ROI_boxes(matlab, image_filepath, method)

    # Load the demo image
    im = cv2.imread(image_filepath)
    scores, boxes = im_detect(net, im, obj_proposals)
    timer.toc()

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls in classes:
        cls_ind = CLASSES.index(cls)
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        keep = np.where(cls_scores >= CONF_THRESH)[0]
        cls_boxes = cls_boxes[keep, :]
        cls_scores = cls_scores[keep]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
    return dets, timer.total_time


def initialize(cmd):
    if cmd == 'ss':
        model_dirname = '20160809_SS_train0.8'
    elif cmd == 'ed':
        model_dirname = '20160808_EAR0.4.2_train0.8'
    elif cmd == 'BING':
        model_dirname = '20160807_BING800_train0.8'
    else:
        raise IOError('Wrong cmd name, choose from ss, ed, BING')
    # configuration for the caffe net
    prototxt = os.path.join(cfg.ROOT_DIR, 'models', NETS['caffenet'][0],
                            'test.prototxt')
    caffemodel = os.path.join(cfg.ROOT_DIR, 'output', model_dirname, 'soton_ear',
                              NETS['caffenet'][1])
    caffe.set_mode_gpu()
    caffe.set_device(0)
    net = caffe.Net(prototxt, caffemodel, caffe.TEST)
    print '\nLoaded network {:s}'.format(caffemodel)

    # initialize the MATLAB server
    print '\nMATLAB Connected'
    import matlab_wrapper
    matlab = matlab_wrapper.MatlabSession()
    print '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'

    return net, matlab


def visualise(image_index,image_filepath, dets,ratio):
    """Draw detected bounding boxes."""
    (x1, y1, x2, y2) = get_gt(image_index)
    im = cv2.imread(image_filepath)
    cv2.namedWindow('frame')
    if len(dets) == 0:
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(im, 'Fail to detec ear in this image',
                    (50, 50), font, 0.5, (0, 0, 255), 1)
        cv2.imshow('frame',im)
        cv2.waitKey(10)
        return
    for bbox in dets:
        cv2.rectangle(im, (x1, y1),(x2,y2),(0,255,0),2)
        cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(im, 'IOU ratio {:.3f}'.format(max(ratio)),
                    (int(bbox[0]), int(bbox[1] - 10)), font, 0.5, (0, 0, 255), 1)
    cv2.imshow('frame', im)
    cv2.waitKey(10)

def result(dets, image_index,image_filepath, method, time):
    ratio = IOU_ratio(image_index, dets)
    visualise(image_index,image_filepath,dets,ratio)
    global false_count, count, false_positive_count
    count += 1
    if (len(dets) == 0):
        false_count += 1
    elif (max(ratio) <= 0.5):
        false_positive_count +=1
    print('{:d}/{:d}/{:d} fail/total/FP detect by {:s} OP_method at {:.3f} seconds').\
        format(false_count, count, false_positive_count, method, time)

if __name__ == '__main__':
    cmd = 'ed'
    net, matlab = initialize(cmd)
    index_csv_path = os.path.join(cfg.ROOT_DIR, 'ear_recognition', 'data_file', 'test_image_index_list.csv')
    with open(index_csv_path, 'rb') as mycsvfile:
        image_list = csv.reader(mycsvfile)
        for index, item in enumerate(image_list):
            image_filepath = str(item[0])
            # image_filepath = transform_image(index + 1, 'occlude', 0.5)
            dets, time = demo(net, matlab, image_filepath, ('ear',), cmd)
            result(dets, index+1, image_filepath, cmd, time)
    print('Total {} images {} fails'.format(count, false_count))
    # transform_image(1, 'occlude', 0.1)
