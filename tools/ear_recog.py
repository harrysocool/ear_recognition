import csv
import linecache
import _init_paths
from lib.fast_rcnn.config import cfg
from lib.fast_rcnn.test import im_detect
from lib.utils.nms import nms
from lib.utils.timer import Timer
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import pandas as pd
from OP_methods.BING_Objectness.source.bing_demo import bing_demo

OP_num = 0
CLASSES = ('__background__', 'ear')

NETS = {'vgg16': ('VGG16',
                  'vgg16_fast_rcnn_iter_40000.caffemodel'),
        'vgg_cnn_m_1024': ('VGG_CNN_M_1024',
                           'vgg_cnn_m_1024_fast_rcnn_iter_40000.caffemodel'),
        'caffenet': ('CaffeNet',
                     'caffenet_fast_rcnn_iter_40000.caffemodel')}



class cmd_result(object):
    def __init__(self, cmd):
        self.cmd = cmd
        self.count = 0
        self.false_count = 0
        self.false_positive_count = 0
        self.true_ratio = []

    def gather(self):
        self.true_ratio.append(['Total pictures', self.count])
        self.true_ratio.append(['True negative', self.false_count])
        self.true_ratio.append(['False positive', self.false_positive_count])


def IOU_ratio(image_index, dets):
    ratio = []
    (x1, y1, x2, y2) = get_gt(image_index)
    for box in dets:
        X1 = box[0]
        Y1 = box[1]
        X2 = box[2]
        Y2 = box[3]
        if (X1 <= np.float32(x1) and X2 >= np.float32(x2)
            and Y1 <= np.float32(y1) and Y2 >= np.float32(y2)):
            ratio.append(1.0)
        else:
            SI = max(0, min(x2, X2) - max(x1, X1)) * \
                 max(0, min(y2, Y2) - max(y1, Y1))
            SU = (x2 - x1) * (y2 - y1) + (X2 - X1) * (Y2 - Y1) - SI
            ratio.append(SI / SU)
    if len(ratio) == 0:
        max_ratio = 0
    else:
        max_ratio = max(ratio)
    return max_ratio


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
        # matlab.eval("toolboxCompile")
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

    if variable == 0:
        return image_filepath
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
        gauss = np.random.normal(0, variable, im.shape)
        gauss = gauss.reshape(im.shape)
        noisy = im + gauss

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
    global OP_num
    OP_num = len(obj_proposals)
    if len(obj_proposals)==0:
        dets = []
        timer.toc()
        return dets, timer.total_time

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


def visualise(image_index, image_filepath, dets, ratio):
    """Draw detected bounding boxes."""
    (x1, y1, x2, y2) = get_gt(image_index)
    im = cv2.imread(image_filepath)
    cv2.namedWindow('frame')
    if len(dets) == 0:
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(im, 'Fail to detect ear in this image',
                    (50, 50), font, 0.5, (0, 0, 255), 1)
        cv2.imshow('frame', im)
        cv2.waitKey(10)
        return
    for bbox in dets:
        cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(im, 'IOU ratio {:.3f}'.format(ratio),
                    (int(bbox[0]), int(bbox[1] - 10)), font, 0.5, (0, 0, 255), 1)
    cv2.imshow('frame', im)
    cv2.waitKey(10)


def result(object1, dets, image_index, image_filepath, method, time):

    ratio = IOU_ratio(image_index, dets)

    # visualise(image_index, image_filepath, dets, ratio)

    object1.count += 1
    if (len(dets) == 0):
        object1.false_count += 1
    elif (ratio <= 0.5):
        object1.false_positive_count += 1
    else:
        object1.true_ratio.append(ratio)
    print('{:d}/{:d}/{:d} fail/total/FP detect by {:s} OP_method at {:.3f} seconds, {} OP boxes'). \
        format(object1.false_count, object1.count, object1.false_positive_count, method, time, OP_num)


def save_result(list1, image_filepath, cmd, transform, variable):
    dir_path = os.path.dirname(image_filepath)
    dir_path1 = os.path.dirname(dir_path)
    csv_file_name = os.path.join(dir_path1,'result',
                                 cmd + '_' + transform +'_'+str(variable)+'.csv')
    temp = pd.DataFrame(list1)
    temp.to_csv(csv_file_name, index=False, header=False)


if __name__ == '__main__':
    OP_method = ('ed','ss', 'BING')
    transform_prod = ('noise', 'occlude')
    variable_prod = ((0,5,10,15,20,25,30),(0.1,0.2,0.3,0.4,0.5))

    cmd = OP_method[2]
    transform = transform_prod[1]
    variable = variable_prod[1][4]

    index_csv_path = os.path.join(cfg.ROOT_DIR,
                                  'ear_recognition', 'data_file', 'test_image_index_list.csv')

    net, matlab = initialize(cmd)
    object1 = cmd_result(cmd)

    with open(index_csv_path, 'rb') as mycsvfile:
        image_list = csv.reader(mycsvfile)
        for index, item in enumerate(image_list):
            image_filepath = str(item[0])
            image_filepath = transform_image(index + 1, transform, variable)

            dets, time = demo(net, matlab, image_filepath, ('ear',), cmd)
            result(object1, dets, index + 1, image_filepath, cmd, time)

    object1.gather()
    save_result(object1.true_ratio, image_filepath, cmd, transform, variable)

