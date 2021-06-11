
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer  # 多标签编码
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score
# from transformers import AlbertConfig, AlbertTokenizer, TFAlbertModel
from transformers import BertConfig,BertTokenizer,TFBertModel,TFBertForSequenceClassification
import tensorflow as tf
import os
# from metric import Metrics
from tensorflow.keras.layers import Dropout, Dense, LSTM
import sys
import json
import random
from keras import backend as K

model_name='bert-base-multilingual-uncased'
dev_file = "../Data/dev.csv"
train_file = "../Data/train.csv"
# test_file = "../Data/test.txt"
test_file='../Data/dev_as_test.csv'
task_output_file = "./task4_result/bert_multilabel_result.csv"
checkpoint_save_path = "./Model/bert_muti_labels.h5"

def load_test_data():
    data_df = pd.read_csv(test_file)

    id, text = [], []

    for idx in data_df['id']:
        id.append(idx)
    for tex in data_df['text']:
        text.append(tex)

    # print("load test data")
    return id, text


def load_train_data():
    data_df = pd.read_csv(train_file)

    id, text, labels = [], [], []
    for idx in data_df['id']:
        id.append(idx)
    for tex in data_df['text']:
        text.append(tex)
    for la in data_df['humor_target']:
        labels.append(la)

    # print("load train data")
    return id, text, labels


def load_dev_data():
    data_df = pd.read_csv(dev_file)

    id, text, labels = [], [], []
    for idx in data_df['id']:
        id.append(idx)
    for tex in data_df['text']:
        text.append(tex)
    for la in data_df['humor_target']:
        labels.append(la)

    # print("load dev data")
    return id, text, labels


def save_result(test_labels,test_file,task_output_file):
  df_test = pd.read_csv(test_file)  # 读测试集的text 一并在结果中输出
  ids_test = df_test['id'].tolist()
  text_test = df_test['text'].tolist()

  s=[]
  for i in range(len(test_labels)):
      # print(a[i])
    str=""
    for j in range(len(test_labels[i])):
        if j == 0:
            str += test_labels[i][j].strip(",")
        else:
            str += (';'+ test_labels[i][j]).strip(",")
    s.append(str)

  # 将预测结果 写回task2_humor_rating文件中
  print('writing test result to task4_result')
  pd.DataFrame({'id': ids_test, 'text': text_test, 'humor_target': s}).to_csv(task_output_file, index=False)


class Metrics(tf.keras.callbacks.Callback):
    def __init__(self, valid_data):
        super(Metrics, self).__init__()
        self.validation_data = valid_data

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_predict = self.model.predict(self.validation_data[0])
        # val_predict = np.argmax(self.model.predict(self.validation_data[0]), -1)
        print(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        print(val_predict[20])

        threshold, upper, lower = 0.5, 1, 0
        val_predict[val_predict > threshold] = upper
        val_predict[val_predict <= threshold] = lower

        val_predict = tf.cast(val_predict, dtype=tf.int32)
        val_predict = val_predict.numpy()

        print("prdic:", val_predict[20])

        val_targ = self.validation_data[1]
        print("ture:", val_targ[20])
        print(
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")

        _val_f1_macro = f1_score(val_targ, val_predict, average='macro')
        _val_f1_micro = f1_score(val_targ, val_predict, average='micro')
        _val_f1 = f1_score(val_targ, val_predict, average='macro')
        _val_precision = precision_score(val_targ, val_predict, average='macro')
        logs['val_f1_macro'] = _val_f1_macro
        logs['val_f1_micro'] = _val_f1_micro
        logs['val_f1'] = _val_f1
        logs['val_precision'] = _val_precision
        # print(_val_f1)
        print("val_precision:  ", _val_precision)
        print(" — val_f1_macro: %f  — val_f1_Micro: %f  — val_f1: %f" % (_val_f1_macro, _val_f1_micro, _val_f1))
        return


def f1(y_true, y_pred):
    # f1值作为评估参数
    def recall(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


def create_model():
    # bert层
    config = BertConfig.from_pretrained(model_name, output_attentions=True)
    print(config)
    bert_layer = TFBertModel.from_pretrained(model_name)
    initializer = tf.keras.initializers.TruncatedNormal(config.initializer_range)

    # 构建bert输入
    input_ids = tf.keras.Input(shape=(config.max_position_embeddings,), dtype='int32')
    token_type_ids = tf.keras.Input(shape=(config.max_position_embeddings,), dtype='int32')
    attention_mask = tf.keras.Input(shape=(config.max_position_embeddings,), dtype='int32')
    inputs = {'input_ids': input_ids, 'token_type_ids': token_type_ids, 'attention_mask': attention_mask}

    # bert的输出
    bert_output = bert_layer(inputs)
    # print(bert_output)
    hidden_states = bert_output[1]
    # print(hidden_states)

    dropout_hidden = tf.keras.layers.Dropout(0.3)(hidden_states)

    dense = tf.keras.layers.Dense(768, activation='relu')(dropout_hidden)
    dropout = tf.keras.layers.Dropout(0.4)(dense)
    output = tf.keras.layers.Dense(15, kernel_initializer=initializer, activation='sigmoid')(dropout)
    # output = output_layer(initializer)(dropout_output)
    # print(output)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    # print(config.max_position_embeddings)
    model.summary()
    return model

# 读取labels列表
labels_list_filename = '../Data/labels_list_task4.txt'
with open(labels_list_filename, "r") as f:
    labels_list = [line.rstrip() for line in f.readlines()]
print(labels_list)

# load data
train_id, train_texts, train_labels = load_train_data()
dev_id, dev_texts, dev_labels = load_dev_data()
test_id, test_texts = load_test_data()

print("train data size:", len(train_texts))
print("val data size:", len(dev_texts))
print("test data size:", len(test_texts))

# 对标签进行编码
print("tec",labels_list)
mlb = MultiLabelBinarizer(classes=labels_list)
# train_labels = mlb.fit_transform(train_labels)
# dev_labels = mlb.fit_transform(dev_labels)

new_train_label = []
for train_lab in train_labels:
  tlb=train_lab.replace("; ",";")
  new_train_label.append(tlb.split(';'))
# print("train",new_train_label)
train_labels = mlb.fit_transform(new_train_label)

new_dev_label = []
for dev_lab in dev_labels:
  dlb=dev_lab.replace("; ",";")
  new_dev_label.append(dlb.split(';'))
# print("dev",new_dev_label)
dev_labels = mlb.fit_transform(new_dev_label)


# 数据编码
tokenizer = BertTokenizer.from_pretrained(model_name)
train_encodings = tokenizer(train_texts, truncation=True, padding='max_length')
dev_encodings = tokenizer(dev_texts, truncation=True, padding='max_length')
test_encodings = tokenizer(test_texts, truncation=True, padding='max_length')
# print(train_encodings.keys())


train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    train_labels
))

val_dataset = tf.data.Dataset.from_tensor_slices((
    dict(dev_encodings),
    dev_labels
))

# 先用划分的验证集做测试以计算F1
test_dataset = tf.data.Dataset.from_tensor_slices((
    dict(test_encodings),
))


# 创建网络
model = create_model()
# 断点续训
if os.path.exists(checkpoint_save_path):
    print('================== Loading Model to train ====================')
    model.load_weights(checkpoint_save_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path, monitor='val_f1', mode='max', verbose=1,
                                                 save_weights_only=True, save_best_only=True, )


tb_callback = tf.keras.callbacks.TensorBoard(log_dir='./log/bert_multilingual_Crossentropy_logs', profile_batch=0)

# 训练模型
optimizer = tf.keras.optimizers.Adam(lr=5e-6 ,epsilon=1e-2)
loss = tf.keras.losses.BinaryCrossentropy()  # 二进制交叉熵
# loss = tf.keras.losses.CategoricalCrossentropy()
# metric = tf.keras.metrics.CategoricalAccuracy()
# metric = tf.keras.metrics.BinaryAccuracy()
metric = f1

model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
# model.compile(optimizer=optimizer, loss=tfa.losses.WeightedKappaLoss(num_classes=12), metrics=['accuracy'])
# model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

print("===============Start training=================")
history = model.fit(train_dataset.shuffle(1000).batch(8), validation_data=val_dataset.batch(8), epochs=200,
                    batch_size=8,
                    callbacks=[Metrics(valid_data=(val_dataset.batch(8), dev_labels)),
                               cp_callback,
                               tb_callback,
                               tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=50, verbose=1)
                               ]
                    )

# 预测结果
# model = create_model()
model.load_weights(checkpoint_save_path)  # 取val最优模型
print("=======================loading model to eval ======================")
y_pred = model.predict(val_dataset.batch(8))
y_true = dev_labels

print(len(y_pred),len(y_true))

# 求F1
import copy
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
y_pred_t = copy.deepcopy(y_pred)
threshold = 0.5
upper, lower = 1, 0
y_pred_t[y_pred_t > threshold] = upper
y_pred_t[y_pred_t <= threshold] = lower

y_pred_t = tf.cast(y_pred_t, dtype=tf.int32)
y_pred_t = y_pred_t.numpy()

F1_Macro = f1_score(y_true, y_pred_t, average='macro')
print("F1 macro:",F1_Macro)

# 在test上预测
if os.path.exists(checkpoint_save_path):
    print('================Loading Model to testing =================')
    model.load_weights(checkpoint_save_path)

y_pred = model.predict(test_dataset.batch(8))
threshold, upper, lower = 0.5, 1, 0
y_pred[y_pred > threshold] = upper
y_pred[y_pred <= threshold] = lower

y_pred = tf.cast(y_pred, dtype=tf.int32)
y_pred = y_pred.numpy()

# 将结果转为labels list形式
test_labels = mlb.inverse_transform(y_pred)

# 保存预测结果
save_result(test_labels,test_file,task_output_file)









