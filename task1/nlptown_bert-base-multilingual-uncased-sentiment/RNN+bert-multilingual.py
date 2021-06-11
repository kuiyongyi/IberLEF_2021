# -*- coding: utf-8 -*-
# pip install transformers

import os
import csv
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data as Data
from sklearn.utils import shuffle as reset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW,AutoModel
from tqdm import tqdm
import transformers
from transformers import *
# from conv import text_cnn
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

transformers.logging.set_verbosity_error()

class_id = ['0', '1']

model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
output_model = './Models/model.pth'
best_score = 0  # 初始化 最好的得分为0
batch_size = 64
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_path = '../Data/train.csv'
dev_path = '../Data/dev.csv'
# test_path = '../Data/test.csv'
test_path = '../Data/dev_as_test.csv'
result_path = './task1_result/RNN-bert_task1_classify_result.csv'


def save(model, optimizer):
    # save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, output_model)
    print('The best model has been saved')  # 保存最好的结果的模型


class CustomDataset(Data.Dataset):
    # with_labels则表示在 训练、验证时
    def __init__(self, data, maxlen, with_labels=True, model_name=model_name):
        self.data = data  # pandas dataframe

        # 初始化tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.maxlen = maxlen
        self.with_labels = with_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        # 在数据帧中指定的索引处选择sentence1和sentence2 本次仅仅涉及sentence1
        sent = str(self.data.loc[index, 'text'])

        # 对这对句子进行标记，以获得标记id、注意掩码和标记类型id
        encoded_pair = self.tokenizer(sent,
                                      padding='max_length',
                                      truncation=True,  # 以max_length进行截断
                                      max_length=self.maxlen,
                                      return_tensors='pt')  # 返回 torch.Tensor objects

        token_ids = encoded_pair['input_ids'].squeeze(0)  # token ids的张量

        # 二进制张量，填充值为“0”，其他值为“1”
        attn_masks = encoded_pair['attention_mask'].squeeze(0)

        # 第1句标记为“0”，第2句标记为“1”的二元张量
        token_type_ids = encoded_pair['token_type_ids'].squeeze(0)

        if self.with_labels:  # 如果数据集有标签，则为True
            label = self.data.loc[index, 'is_humor']
            return token_ids, attn_masks, token_type_ids, label  # 训练 和 验证时
        else:
            return token_ids, attn_masks, token_type_ids  # 测试时

class MyModel(nn.Module):
    # num_classes表示分类数 默认为2
    def __init__(self, freeze_bert=False, model_name=model_name, hidden_size=768, num_classes=2):
        super(MyModel, self).__init__()
        # 模型bert层
        self.bert = AutoModelForSequenceClassification.from_pretrained(model_name, output_hidden_states=True,return_dict=True)

        if freeze_bert:  # 冻结 预训练参数 只更新下游参数
            for p in self.bert.parameters():
                p.requires_grad = False
        
        self.lstm = nn.LSTM(hidden_size*4, hidden_size, 4, bidirectional=False, batch_first=True, dropout=0.3)
        # self.lstm = nn.LSTM(hidden_size*4, hidden_size, 8 ,dropout=0.3)
        self.maxpool = nn.MaxPool1d(32)

        # 全连接层 
        self.fc = nn.Sequential(
            # nn.Dropout(p=0.4),
            # nn.Linear(hidden_size*2, 192, bias=False),
            # nn.ReLU(),

            # nn.Dropout(p=0.3),
            # nn.Linear(192,num_classes,bias = False),
            # nn.Softmax(),
            
            nn.Dropout(p=0.5),
            nn.Linear(hidden_size, num_classes, bias=False),
            # nn.Softmax(),
        )

    # 前向传播
    def forward(self, input_ids, attn_masks, token_type_ids):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attn_masks)
        # print(len(outputs))

        hidden_states = torch.cat(tuple([outputs.hidden_states[i] for i in [-1, -2, -3, -4]]),
                                  dim=-1)  # [bs, seq_len, hidden_dim*4]--32 192 3072
        out, _ = self.lstm(hidden_states)
        # print(out.shape)
        out = nn.ReLU()(out)
        # print(out.shape)   #[bs, seq_len, hidden_dim*4]--32 192 1536
     

        first_hidden_states = out[:, 0, :]  
        # first_hidden_states = hidden_states[:, 0, :]  # [bs, hidden_dim*4] 32 3072
        # print(first_hidden_states.shape)  # [bs, hidden_dim*4] 32 1536

        logits = self.fc(first_hidden_states)  #bs class_num 32 2
        
        #print(logits.shape)  #bs num_classes - 32 2
        return logits



def set_seed(seed):  # 设置随机种子
    # 设置随机种子，使结果可重现
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def process_data(filename):
    data = pd.read_csv(filename, encoding='utf-8')
    return data


def flat_accuracy(preds, labels):  # 计算精度
    pred_flat = np.argmax(preds, axis=1).flatten()  # [3, 5, 8, 1, 2, ....]
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def f1_SCORE(y_pred, y_true):
    pred_la = np.argmax(y_pred, axis=1).flatten()
    true_labels = y_true.flatten()
    import copy

    y_pred_t = copy.deepcopy(pred_la)
    F1_Macro = f1_score(true_labels, y_pred_t, average='macro')
    F1_Micro = f1_score(true_labels, y_pred_t, average='micro')
    # f1score = f1_score(true_labels, y_pred_t, average='binary')

    return F1_Macro,F1_Micro


def train_test_split(data_df, test_size=0.2, shuffle=True, random_state=None):  # 划分训练集和验证集

    if shuffle:
        data_df = reset(data_df, random_state=random_state)  # 临界值

    train = data_df[int(len(data_df) * test_size):].reset_index(drop=True)
    test = data_df[:int(len(data_df) * test_size)].reset_index(drop=True)

    return train, test


# 验证
def eval_model(model, optimizer, validation_dataloader):
    # model.eval()
    eval_loss, eval_accuracy, nb_eval_steps = 0, 0, 0
    eval_f1_mi, eval_f1_ma,eval_F1,eval_f1 = 0, 0 ,0 ,0

    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            logits = model(batch[0], batch[1], batch[2])
            logits = logits.detach().cpu().numpy()
            label_ids = batch[3].cpu().numpy()

            tmp_eval_accuracy = flat_accuracy(logits, label_ids)

            F1_ma,F1_mi = f1_SCORE(logits,label_ids)

            eval_accuracy += tmp_eval_accuracy
            eval_f1_ma += F1_ma
            eval_f1_mi += F1_mi

            nb_eval_steps += 1

    # f1_SCORE(logits, label_ids)
    # print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))
    print("Validation Accuracy: {}  ".format(eval_accuracy/nb_eval_steps))
    print("Validation---- F1_Macro: {}  F1_Micro: {}  ".format(eval_f1_ma / nb_eval_steps, eval_f1_mi / nb_eval_steps))

    global best_score
    if (best_score < (eval_f1_ma / nb_eval_steps + eval_f1_mi / nb_eval_steps)/2 ):  # 如果当前得到的 模型精度更高 则保存模型
        best_score = (eval_f1_ma / nb_eval_steps + eval_f1_mi / nb_eval_steps)/2
        save(model, optimizer)


# epochs为训练轮数 默认值设为1
def train_eval(model, criterion, optimizer, train_loader, val_loader, epochs=1):
    print('loading model...')
    checkpoint = torch.load(output_model, map_location='cpu')  # 设置断点 断点续训
    model.load_state_dict(checkpoint['model_state_dict'])  # 加载断点的模型
    model.to(device)

    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # 加载优化器

    print('-----Training-----')
    for epoch in range(epochs):
        model.train()  # 训练

        print('Epoch', epoch + 1)
        for i, batch in enumerate(tqdm(train_loader)):
            batch = tuple(t.to(device) for t in batch)
            logits = model(batch[0], batch[1], batch[2])
            loss = criterion(logits, batch[3])
            print(i, "loss: ",loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # model.eval()
            if i % 10 == 0:
              eval_model(model, optimizer, val_loader)

def dev_eval(model, optimizer, validation_dataloader):
    # model.eval()
    print('loading model...')
    checkpoint = torch.load(output_model, map_location='cpu')  # 设置断点 断点续训
    model.load_state_dict(checkpoint['model_state_dict'])  # 加载断点的模型
    model.to(device)

    eval_loss, eval_accuracy, nb_eval_steps = 0, 0, 0
    eval_f1_mi, eval_f1_ma,eval_F1,eval_f1 = 0, 0 ,0 ,0

    for batch in validation_dataloader:
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            logits = model(batch[0], batch[1], batch[2])
            logits = logits.detach().cpu().numpy()
            label_ids = batch[3].cpu().numpy()

            tmp_eval_accuracy = flat_accuracy(logits, label_ids)

            F1_ma,F1_mi = f1_SCORE(logits,label_ids)

            eval_accuracy += tmp_eval_accuracy
            eval_f1_ma += F1_ma
            eval_f1_mi += F1_mi

            nb_eval_steps += 1

    # print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))
    print("Validation Accuracy: {}  ".format(eval_accuracy/nb_eval_steps))
    print("Validation---- F1_Macro: {}  F1_Micro: {}  ".format(eval_f1_ma / nb_eval_steps, eval_f1_mi / nb_eval_steps))

# 测试
def test(model, dataloader, with_labels=False):
    # 加载模型
    print("loading model...")
    checkpoint = torch.load(output_model)

    model.load_state_dict(checkpoint['model_state_dict'])  # 加载优化器
    model.to(device)

    print('-----Testing-----')

    pred_label = []  # 以后将预测结果 存入列表中

    model.eval()

    for i, batch in enumerate(tqdm(dataloader)):
        batch = tuple(t.to(device) for t in batch)
        with torch.no_grad():
            logits = model(batch[0], batch[1], batch[2])
            logits = logits.detach().cpu().numpy()
            # logits = logits.detach()
            # argmax表示将结果预测为 所得概率最大的那一个
            preds = np.argmax(logits, axis=1).flatten()
            pred_label.extend(preds)  # 将预测结果 存入列表中

    # 将预测结果 写回pred文件中
    # pd.DataFrame(data=pred_label, index=range(len(pred_label))).to_csv(result_path, encoding='utf-8')

    import pandas as pd
    df_test = pd.read_csv(test_path)  # 读测试集的text 一并在结果中输出
    ids_test = df_test['id'].tolist()
    text_test = df_test['text'].tolist()
    # 将预测结果 写回task2_humor_rating文件中
    pd.DataFrame({'id': ids_test, 'is_humor': pred_label}).to_csv(result_path, index=False)

    print('Test Completed')


if __name__ == '__main__':
    set_seed(100)  # Set all seeds to make results reproducible
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 创建模型
    model = MyModel(freeze_bert=False, model_name=model_name, hidden_size=768, num_classes=2)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=1e-2)

    train_df = process_data(train_path)

    # 划分训练集 验证集
    # train_df, val_df = train_test_split(data_df, test_size=0.2, shuffle=True, random_state=1)

    print("Reading training data...")  # 训练数据
    train_set = CustomDataset(train_df, maxlen=96, model_name=model_name)
    train_loader = Data.DataLoader(train_set, batch_size=batch_size, num_workers=2, shuffle=True)
    print('------------------------------------------')

    val_df = process_data(dev_path)
    print("Reading validation data...")  # 验证数据
    val_set = CustomDataset(val_df, maxlen=96, model_name=model_name)
    val_loader = Data.DataLoader(val_set, batch_size=batch_size, num_workers=2, shuffle=True)

    ## 训练及验证
    train_eval(model, criterion, optimizer, train_loader, val_loader, epochs=1)  # 训练轮数

    # print('------------------------------------------')
    # print("Reading test data...")  # 测试数据
    # test_df = process_data(test_path)
    # test_set = CustomDataset(test_df, maxlen=96, with_labels=False, model_name=model_name)
    # test_loader = Data.DataLoader(test_set, batch_size=batch_size, num_workers=2, shuffle=False)

    # # 调用函数 进行测试
    # test(model, test_loader, with_labels=False)
    # print('------------------------------------------')


    # set_seed(100)  # Set all seeds to make results reproducible
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # model = MyModel(freeze_bert=False, model_name=model_name, hidden_size=768, num_classes=2)
    # model.to(device)
    # optimizer = AdamW(model.parameters(), lr=1e-5, weight_decay=1e-2)
 
    val_df = process_data(dev_path)
    print("Reading validation data...")  # 验证数据
    val_set = CustomDataset(val_df, maxlen=96, model_name=model_name)
    val_loader = Data.DataLoader(val_set, batch_size=batch_size, num_workers=10, shuffle=True)

    dev_eval(model,optimizer,val_loader)

    print('------------------------------------------')
    print("Reading test data...")  # 测试数据
    test_df = process_data(test_path)
    test_set = CustomDataset(test_df, maxlen=96, with_labels=False, model_name=model_name)
    test_loader = Data.DataLoader(test_set, batch_size=batch_size, num_workers=10, shuffle=False)

    # 调用函数 进行测试
    test(model, test_loader, with_labels=False)
    print('------------------------------------------')


