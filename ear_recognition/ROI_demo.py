# -*- coding: utf-8 -*-
"""
Created on 16/6/25 21:48 2016

@author: harry sun
"""
if __name__ == '__main__':
    from OP_methods.selective_search_ijcv_with_python import get_windows
    import numpy as np
    from PIL import Image, ImageDraw

    # image_path_list = ['../selective_search_ijcv_with_python/000015.jpg']
    image_path_list = ['../2.jpg']

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
