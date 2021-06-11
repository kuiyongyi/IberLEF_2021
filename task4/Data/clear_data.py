# -*- coding: utf-8 -*-
import csv
import pandas as pd
import sys
import matplotlib.pyplot as plt


import re
def Rep(text):  #对要提取的数据 依次正则 清理数据
    # p = re.compile(r'(<span.*>.*</span>)*(<a.*>.*</ a>>)?')
    # text = re.sub('[^\u4e00-\u9fa5]', '', text)
    # text = p.sub(r'', text)

    pattern = re.compile(r'<[^>]+>', re.S) #去掉HTML标签
    text = pattern.sub('', text)

    pattern1 = re.compile("[^(\u2E80-\u9FFF\\w\\s`~!@#\\$%\\^&\\*\\(\\)_+-？（）——=\\[\\]{}\\|;。，、《》”：；“！……’:'\"<,>\\.?/\\\\*)]")
    text = pattern1.sub('', text)

    text=re.sub('—','',text) #  将#号或【号替换为空
    text = re.sub('- ¿', '', text) # 将】号替换为逗号
    text = re.sub('¿', '', text) #将@的人名去掉
    text = re.sub('#', '', text) 

    # \\会转义成反斜杠, 反斜杠本身就是转义符, 所有就成了“\ | ”, 在进行转义就是 |, 所以\\ | 实际上是“ | ”
    text=re.sub('@','',text) #将|符号去掉
    text = re.sub('"', '', text)
    text = re.sub('¡', '', text) #将",，替换为句号
    text = re.sub('\n','',text)
    text = re.sub('–','',text)
    text = re.sub('-','',text)
    text = re.sub(' -','',text)
    text = re.sub('☄️','',text)
    text = re.sub('!!!!','',text)
    return text  #将干净的数据返回出去


path='/content/drive/MyDrive/IberLEF_HAHA_2021/task4/Data/train.csv'
data=pd.read_csv(path)
ids = data['id'].tolist()
label = data['humor_target'].tolist()

duan=[]
lens = []
for t in data['text']:
  duan.append(Rep(t))
  lens.append(len(t.split()))

plt.hist(lens)
plt.show()

pd.DataFrame({'id': ids, 'text': duan, 'humor_target': label}).to_csv(path, index=False)
