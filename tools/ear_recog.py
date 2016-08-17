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
from ear_recog_demo import ROI_boxes


count = 0
false_count = 0
CLASSES = ('__background__', 'ear')

NETS = {'vgg16': ('VGG16',
                  'vgg16_fast_rcnn_iter_40000.caffemodel'),
        'vgg_cnn_m_1024': ('VGG_CNN_M_1024',
                           'vgg_cnn_m_1024_fast_rcnn_iter_40000.caffemodel'),
        'caffenet': ('CaffeNet',
                     'caffenet_fast_rcnn_iter_40000.caffemodel')}

OP_method = ('ss','ed','BING')

def transform_image(image_index, cmd, variable=None):
    assert variable!=None, 'No variable input, needs more variable input'
    gt_csv_path = os.path.join(cfg.ROOT_DIR, 'ear_recognition', 'data_file', 'test_gt_roidb.csv')
    index_csv_path = os.path.join(cfg.ROOT_DIR, 'ear_recognition', 'data_file', 'test_image_index_list.csv')
    image_filepath = linecache.getline(index_csv_path, image_index).strip('\n')

    # craete the folder for transform iamge
    datasets_path = '/home/harrysocool/Github/fast-rcnn/DatabaseEars/'
    file_name = os.path.basename(image_filepath)
    dir_path = os.path.join(datasets_path, cmd +'_'+ str(variable))
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
        noise = np.random.randn(im.shape)*variable
        noise_im = float_im + noise
        noisy = np.uint8(np.clip(noise_im, 0, 255))
        cv2.imwrite(new_image_filepath, noisy)
    elif cmd == 'occlude':
        line = linecache.getline(gt_csv_path, image_index).strip('\n').split()
        x1 = float(line[-4])
        y1 = float(line[-3])
        x2 = float(line[-2])
        y2 = float(line[-1])
        new_y2 = round(y1+(y2-y1) * variable)
        mask = np.ones(im.shape[:2], np.uint8)
        mask[y1:new_y2,x1:x2] = 0
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
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        keep = np.where(cls_scores >= CONF_THRESH)[0]
        cls_boxes = cls_boxes[keep, :]
        cls_scores = cls_scores[keep]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        global false_count, count
        count += 1
        if (len(dets) == 0):
            false_count += 1
        print('{:d}/{:d} fail detect by {:s} OP_method at {:.3f} seconds').format(false_count, count,
                                                                                         method, timer.total_time)


def initialize(cmd):
    if cmd is 'ss':
        model_dirname = '20160809_SS_train0.8'
    elif cmd is 'ed':
        model_dirname = '20160808_EAR0.4.2_train0.8'
    elif cmd is 'BING':
        model_dirname = '20160807_BING800_train0.8'
    else:
        raise IOError('Wrong cmd name, choose from ss, ed, BING')
    # configuration for the caffe net
    prototxt = os.path.join(cfg.ROOT_DIR, 'models', NETS['caffenet'][0],
                            'test.prototxt')
    caffemodel = os.path.join(cfg.ROOT_DIR, 'output', model_dirname , 'soton_ear',
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

if __name__ == '__main__':
    cmd = 'ss'
    net, matlab = initialize(cmd)
    index_csv_path = os.path.join(cfg.ROOT_DIR, 'ear_recognition', 'data_file', 'test_image_index_list.csv')
    with open(index_csv_path, 'rb') as mycsvfile:
        image_list = csv.reader(mycsvfile)
        for index, item in enumerate(image_list):
            image_filepath = str(item[0])
            image_filepath = transform_image(index+1, 'occlude', 0.5)
            demo(net, matlab, image_filepath, ('ear',), cmd)
    print('Total {} images {} fails'.format(count, false_count))
    # transform_image(1, 'occlude', 0.1)