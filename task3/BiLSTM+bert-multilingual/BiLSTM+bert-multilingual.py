import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer  # 多标签编码
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score
from transformers import BertConfig, BertTokenizer, TFBertModel, TFBertForSequenceClassification
import tensorflow as tf
import os
# from metric import f1, Metrics
from tensorflow.keras.layers import Dropout, Dense, LSTM
import sys
import json
import random
from keras import backend as K

model_name = 'bert-base-multilingual-uncased'
dev_file = "../Data/dev.csv"
train_file = "../Data/train.csv"
# test_file = "../Data/test.txt"
test_file = '../Data/dev_as_test.csv'
task_output_file = "./task3_result/lstm-con_bert_multi_label_result.csv"
checkpoint_save_path = "./Model/LSTM_muti_labels.h5"

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
    for la in data_df['humor_mechanism']:
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
    for la in data_df['humor_mechanism']:
        labels.append(la)

    # print("load dev data")
    return id, text, labels

def save_result(test_labels, test_file, task_output_file):
    df_test = pd.read_csv(test_file)  # 读测试集的text 一并在结果中输出
    ids_test = df_test['id'].tolist()
    text_test = df_test['text'].tolist()

    # 将预测结果 写回文件中
    print('writing test result to task3_result')
    pd.DataFrame({'id': ids_test, 'text': text_test, 'humor_mechanism': test_labels}).to_csv(task_output_file,index=False)


class Metrics(tf.keras.callbacks.Callback):
    def __init__(self, valid_data):
        super(Metrics, self).__init__()
        self.validation_data = valid_data

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_predict = self.model.predict(self.validation_data[0])  # [100,12]
        print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        print(val_predict[20])

        val_p = tf.zeros(tf.constant(val_predict).shape)
        val = np.argmax(val_predict, -1)  # [100,]
        val_p = val_p.numpy()
        for max_id in range(len(val)):
            val_p[max_id][val[max_id]] = 1
        val_predict = val_p

        val_predict = tf.cast(val_predict, dtype=tf.int32)
        val_predict = val_predict.numpy()

        print("prdic:",val_predict[20])

        val_targ = self.validation_data[1]
        print("ture:",val_targ[20])
        print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")

        _val_f1_macro = f1_score(val_targ, val_predict, average='macro')
        _val_f1_micro = f1_score(val_targ, val_predict, average='micro')
        _val_f1 = f1_score(val_targ, val_predict, average='macro')
        _val_precision = precision_score(val_targ, val_predict, average='macro')
        logs['val_f1_macro'] = _val_f1_macro
        logs['val_f1_micro'] = _val_f1_micro
        logs['val_f1'] = _val_f1
        logs['val_precision'] = _val_precision
        # print(_val_f1)
        print("val_precision: "  , _val_precision)
        print(" — val_f1_macro: %f  — val_f1_Micro: %f  — val_f1: %f" % (_val_f1_macro, _val_f1_micro ,_val_f1))
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

LSTM_layer1 = tf.keras.Sequential([
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(192,  return_sequences=True)),
    Dropout(0.5),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,  return_sequences=False)),
    Dropout(0.4),
])

# LSTM_layer1 = tf.keras.Sequential([
#     LSTM(192, return_sequences=True),
#     Dropout(0.3),
#     LSTM(64),
#     Dropout(0.3),
# ])

# LSTM_layer2 = tf.keras.Sequential([
#     LSTM(192, return_sequences=True),
#     Dropout(0.4),
#     LSTM(64),
#     Dropout(0.2),
# ])

# LSTM_layer3 = tf.keras.Sequential([
#     LSTM(256, return_sequences=True),
#     Dropout(0.5),
#     LSTM(128),
#     Dropout(0.1),
# ])

# 设置一个config类，便于参数配置与更改
class Config(dict):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def set(self, key, val):
        self[key] = val
        setattr(self, key, val)

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
    print("ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc")
    hidden_states = bert_output[0]
    # hidden_states2 = bert_output[1]

    dropout_hidden = tf.keras.layers.Dropout(0.2)(hidden_states)
    print(dropout_hidden.shape) #(None, 512, 768)

    output1 = LSTM_layer1(dropout_hidden)
    # output2 = LSTM_layer2(dropout_hidden)
    # output3 = LSTM_layer3(dropout_hidden)

    # output = tf.concat([output1, output2], 1)
    # output = 1 / 3 * (output1 + output2 + output3)
    output = output1
    print(output.shape)  #(None, 128)
    print("ccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc")

    dropout_output = tf.keras.layers.Dropout(0.5)(output)

    # dense = tf.keras.layers.Dense(64, activation='relu')(dropout_output)
    # dropout = tf.keras.layers.Dropout(0.2)(dense)
    output = tf.keras.layers.Dense(12, kernel_initializer=initializer, activation='softmax')(dropout_output)
    model = tf.keras.Model(inputs=inputs, outputs=output)
    model.summary()
    return model


# 读取labels列表
labels_list_filename = '../Data/labels_list_task3.txt'
with open(labels_list_filename, "r") as f:
    tables_list = [line.rstrip() for line in f.readlines() if len(line) > 2]
print(tables_list)

# load data
train_id, train_texts, train_lables = load_train_data()
dev_id, dev_texts, dev_label = load_dev_data()
test_id, test_texts = load_test_data()

print("train data size:", len(train_texts))
print("val data size:", len(dev_texts))
print("test data size:", len(test_texts))

# 对标签进行编码
mlb = MultiLabelBinarizer(classes=tables_list)
# train_labels = mlb.fit_transform(train_lables)
# dev_labels = mlb.fit_transform(dev_label)

new_train_label = []
for train_lab in train_lables:
    new_train_label.append([train_lab])
train_labels = mlb.fit_transform(new_train_label)
print(new_train_label)
print(len(new_train_label))

new_dev_label = []
for dev_lab in dev_label:
    new_dev_label.append([dev_lab])
dev_labels = mlb.fit_transform(new_dev_label)

# 数据编码
tokenizer = BertTokenizer.from_pretrained(model_name)
train_encodings = tokenizer(train_texts, truncation=True, padding='max_length')
dev_encodings = tokenizer(dev_texts, truncation=True, padding='max_length')
test_encodings = tokenizer(test_texts, truncation=True, padding='max_length')


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
tb_callback = tf.keras.callbacks.TensorBoard(log_dir='./log/lstm-con_bert_multilingual_Crossentropy_logs', profile_batch=0)

# 训练模型
optimizer = tf.keras.optimizers.Adam(lr=5e-6 ,epsilon=1e-2)
# loss = tf.keras.losses.BinaryCrossentropy()  # 二进制交叉熵
loss = tf.keras.losses.CategoricalCrossentropy()
# metric = tf.keras.metrics.CategoricalAccuracy()
# metric = tf.keras.metrics.BinaryAccuracy()
metric = f1

model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
# model.compile(optimizer=optimizer, loss=tfa.losses.WeightedKappaLoss(num_classes=12), metrics=['accuracy'])
# model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

print("===============Start training=================")
history = model.fit(train_dataset.shuffle(1000).batch(8), validation_data=val_dataset.batch(8), epochs=1000,
                    batch_size=8,
                    callbacks=[Metrics(valid_data=(val_dataset.batch(8), dev_labels)),
                               cp_callback,
                               tb_callback,
                              #  tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=30, verbose=1)
                               ]
                    )

# 预测结果
model.load_weights(checkpoint_save_path)  # 取val最优模型
print("=======================loading model to eval ======================")
y_pred = model.predict(val_dataset.batch(8))

val_p = tf.zeros(tf.constant(y_pred).shape)
val = np.argmax(y_pred, -1)  # [100,]
val_p = val_p.numpy()
for max_id in range(len(val)):
    val_p[max_id][val[max_id]] = 1
y_pred = val_p

y_pred = tf.cast(y_pred, dtype=tf.int32)
y_pred = y_pred.numpy()

y_true = dev_labels
print(len(y_pred), len(y_true))

# 求F1
import copy
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

y_pred_t = copy.deepcopy(y_pred)
F1_Macro = f1_score(y_true, y_pred_t, average='macro')
print("F1 macro:", F1_Macro)


# 在test上预测
if os.path.exists(checkpoint_save_path):
    print('================Loading Model to testing =================')
    model.load_weights(checkpoint_save_path)

y_pred = model.predict(test_dataset.batch(8))

val_p = tf.zeros(tf.constant(y_pred).shape)
val = np.argmax(y_pred, -1)  # [100,]
val_p = val_p.numpy()
for max_id in range(len(val)):
    val_p[max_id][val[max_id]] = 1
y_pred = val_p

y_pred = tf.cast(y_pred, dtype=tf.int32)
y_pred = y_pred.numpy()

# 将结果转为labels list形式
test_lab = mlb.inverse_transform(y_pred)

test_labels = []
for lab in test_lab:
    for k in lab:
        test_labels.append(k)
# 保存预测结果
save_result(test_labels, test_file, task_output_file)



