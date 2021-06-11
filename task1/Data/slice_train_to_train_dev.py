# -*- coding: utf-8 -*-
import csv
import pandas as pd
import sys

train_data_path='/content/drive/MyDrive/IberLEF_HAHA_2021/haha_2021_train.csv'
data=pd.read_csv(train_data_path)

# len(data)  #24000
count=19200 #24000*0.8  取百分之80 20作为训练集与验证集

del data['votes_no']
del data['votes_1']
del data['votes_2']
del data['votes_3']
del data['votes_4']
del data['votes_5']
del data['humor_rating']
del data['humor_mechanism']
del data['humor_target']

from sklearn.utils import shuffle
data = shuffle(data)

train_pd=data[0:count]
dev_pd=data[count:24000]

train_pd.to_csv('./train.csv', index=False)
dev_pd.to_csv('./dev.csv', index=False)


