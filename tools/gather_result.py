# -*- coding: utf-8 -*-
"""
Created on 16/8/20 11:38 2016

@author: harry sun
"""
import csv
import os
import pandas as pd
import numpy as np

result_path = os.path.dirname(__file__)
result_path1 = os.path.dirname(result_path)
result_path2 = os.path.join(result_path1, 'result')

filename_list = os.listdir(result_path2)

result = {'noise':{},'occlude':{}}

for filename in filename_list:
    filepath = os.path.join(result_path2, filename)
    if not filename.startswith('.'):
        csv_list = pd.read_csv(filepath).values
        name_list = filename.split('_')
        # get the data
        total = float(csv_list[-3][0].strip('[ ]').split(',')[-1])
        false = float(csv_list[-2][0].strip('[ ]').split(',')[-1])
        false_positive = float(csv_list[-1][0].strip('[ ]').split(',')[-1])
        temp = np.asarray(csv_list[0:-3], np.float32)
        DR = (total-false)/total
        if (total-false) != 0:
            FPR = false_positive/(total-false)
        else:
            FPR = 1.0
        mIOU = float(np.mean(temp))
        # get the parameters
        cmd = name_list[0]
        transform = name_list[1]
        variable = name_list[2][0:-4]

        # fit the parameters into the dictionary
        if float(variable) < 1:
            temp_dict = result['occlude']
        elif float(variable) > 1:
            temp_dict = result['noise']

        if not temp_dict.has_key(cmd):
            temp_dict.setdefault(cmd)
            temp_dict[cmd] = {}
        if not temp_dict[cmd].has_key(variable):
            temp_dict[cmd].setdefault(variable)
        temp_dict[cmd][variable] = [DR, FPR, mIOU]

csv_filepath = os.path.join(result_path1, 'noise_result.csv')

result_list = []

t_result = result['noise']
for t_cmd in t_result:
    tt_result = t_result[t_cmd]
    for t_var in tt_result:
        ttt_result = list(tt_result[str(t_var)])
        ttt_result.insert(0, t_var)
        ttt_result.insert(0, t_cmd)
        result_list.append(ttt_result)
temp = pd.DataFrame(result_list)
temp.to_csv(csv_filepath, index=False, header=False)
