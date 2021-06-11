# -*- coding: utf-8 -*-
import csv
import pandas as pd
import sys

train_data_path='/content/drive/MyDrive/IberLEF_HAHA_2021/haha_2021_train.csv'
data=pd.read_csv(train_data_path)

del data['is_humor']
del data['votes_no']
del data['votes_1']
del data['votes_2']
del data['votes_3']
del data['votes_4']
del data['votes_5']
del data['humor_mechanism']
del data['humor_target']

data1=data.dropna(axis=0) #删除rating为空的行
# len(data1) #9253
from sklearn.utils import shuffle
data1 = shuffle(data1)

count=7402 #9253*0.8  取百分之80 20作为训练集与验证集

train_pd=data1[0:count]
dev_pd=data1[count:len(data1)]

train_pd.to_csv('./train.csv', index=False)
dev_pd.to_csv('./dev.csv', index=False)



