# -*- coding: utf-8 -*-
# !pip install transformers

import os
import torch, argparse
import pandas, numpy
import sys
import math

from sklearn.metrics import accuracy_score, mean_squared_error
# from transformers import GPT2ForSequenceClassification, GPT2Tokenizer
from transformers import RobertaTokenizer, RobertaForCausalLM, RobertaConfig, RobertaForSequenceClassification,AutoModelForSequenceClassification
# from transformers import XLMConfig, XLMModel,XLMTokenizer,XLMForSequenceClassification
from transformers import TrainingArguments, Trainer
import csv, random

task = 'humor_rating'
train_path='../Data/train.csv'
dev_path='../Data/dev.csv'
# test_path='../Data/test.csv'
test_path='../Data/dev_as_test.csv'
result_path='./task2_result/task2_humor_rating.csv'

class load_data_task1(torch.utils.data.Dataset):
    def __init__(self, input_file, tokenizer, max_len, task, split):

        self.input_file = input_file
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.task = task
        self.split = split

        self.examples = self.read_file(input_file)

        # cutoff = int(len(self.examples) * .8)

        # if split != 'test':
        #     random.seed(42)
        #     # random.shuffle(self.examples)

        #     if split == 'train':
        #         self.examples = self.examples[:cutoff]
        #     elif split == 'eval':
        #         self.examples = self.examples[cutoff:]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        instance = self.examples[idx]
        if self.split == 'test':
            sent = instance
            enc = self.tokenizer(
                sent,
                max_length=self.max_len,
                truncation=True,
                padding='max_length',
                return_token_type_ids=True,
                return_tensors='pt'
            )

            return {
                'input_ids': enc['input_ids'].squeeze(0),
                'attention_mask': enc['attention_mask'].squeeze(0),
                'token_type_ids': enc['token_type_ids'].squeeze(0),
            }

        sent = instance[0]
        if self.task == 'is_humor':
            label = int(instance[1])
        elif self.task == 'humor_rating':
            label = float(instance[1])

        enc = self.tokenizer(
            sent,
            max_length=self.max_len,
            truncation=True,
            padding='max_length',
            return_token_type_ids=True,
            return_tensors='pt'
        )

        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'token_type_ids': enc['token_type_ids'].squeeze(0),
            'labels': label
        }

    def read_file(self, file):
        inps = []

        with open(file, 'r') as f:
            reader = csv.reader(f)
            for line in reader:
                if self.split == 'test':
                    inps.append(line[1])
                elif self.task == 'humor_rating':
                    if line[2]:
                        inps.append((line[1], line[2]))

        return inps[1:]


# def load_test_data():
#     test_df=pd.read_csv(test_file)
#     id, text = [], []
#     id.append(test_df['id'])
#     text.append(test_df['text'])
#     # print("load dev data")
#     return id, text


# def load_train_data():

#     train_df=pd.read_csv(train_file)

#     id, text, labels = [], [], []

#     id.append(train_df['id'])
#     text.append(train_df['text'])
#     labels.append(train_df['is_humor'])

#     # print("load train data")
#     return id, text, labels

# def load_dev_data():

#     dev_df=pd.read_csv(dev_file)

#     id, text, labels = [], [], []

#     id.append(dev_df['id'])
#     text.append(dev_df['text'])
#     labels.append(dev_df['is_humor'])

#     # print("load train data")
#     return id, text, labels

def metrics_acc(eval_pred):
    labels = eval_pred.label_ids
    preds = eval_pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)

    return {"accuracy": acc}


def metrics_rmse(eval_pred):  # 均方误差 来 衡量损失
    labels = eval_pred.label_ids
    preds = eval_pred.predictions
    mse = mean_squared_error(labels, preds)
    rmse = math.sqrt(mse)
    rmse = numpy.float64(rmse)

    return {"rmse": rmse}

def set_seed(seed):  # 设置随机种子
    # 设置随机种子，使结果可重现
    import numpy as np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

if __name__ == '__main__':

    set_seed(10)  #
    parser = argparse.ArgumentParser()  # 初始化参数

    parser.add_argument('--load_from_checkpoint', type=str)
    parser.add_argument('--continue_training', type=str)
    parser.add_argument('--output_directory', type=str, default="task2_output")
    parser.add_argument('--tokenizer_path', type=str)
    parser.add_argument('--max_len', type=int, default=192)
    parser.add_argument('--max_steps', type=int, default=40000)  # 最大训练次数
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_gpus', type=int, default=1)

    args = parser.parse_args()  # 参数

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    # tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    # tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token


    # 训练集
    train_dataset = load_data_task1(input_file=train_path, tokenizer=tokenizer, max_len=256,
                                    task=task,
                                    split='train')
    print("train_dataset:",len(train_dataset))                                
    # 验证集
    eval_dataset = load_data_task1(input_file=dev_path, tokenizer=tokenizer, max_len=256,
                                   task=task,
                                   split='eval')
    print("eval_dataset:",len(eval_dataset))

    test_dataset = load_data_task1(input_file=test_path, tokenizer=tokenizer, max_len=256,
                                   task=task,
                                   split='test')
    print("test_dataset:",len(test_dataset)
    # 测试集
    # model = GPT2ForSequenceClassification.from_pretrained('gpt2', num_labels=1)
    # model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=1)
    model = AutoModelForSequenceClassification.from_pretrained('roberta-base',num_labels=1)
    model.resize_token_embeddings(len(tokenizer))

    # fix model padding token id
    model.config.pad_token_id = model.config.eos_token_id

    warmup_steps = int(args.max_steps * .05)  # 调整lr的系数

    training_args = TrainingArguments(
        output_dir=args.output_directory,
        max_steps=args.max_steps,  # 线性步骤数，从0到learning_rate。覆盖的任何效果
        per_device_train_batch_size=args.batch_size,
        logging_steps=500,
        save_total_limit=1,
        # evaluate_during_training=True,
        eval_steps=30,  # 用于数据加载的子进程数。0表示数据将被加载到主进程中
        learning_rate=1e-5,
        warmup_steps=warmup_steps,
        load_best_model_at_end=True,
        metric_for_best_model='eval_rmse',
        disable_tqdm=False
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=metrics_rmse,
    )

    # 训练 验证
    trainer.train()

    # 测试 并将预测结果 写回文件
    predictions = trainer.predict(test_dataset)
    humor_rating_preds = [pred[0] for pred in predictions.predictions]

    output_list = []
    for pred in humor_rating_preds:
        temp = {}
        temp['humor_rating'] = pred
        output_list.append(temp)

    out_df = pandas.DataFrame(output_list)
    # out_df.to_csv(index_label='id','task2_humor_rating.csv', )

    import pandas as pd

    df_test = pd.read_csv(test_path)  # 读测试集的text 一并在结果中输出
    ids_test = df_test['id'].tolist()
    text_test = df_test['text'].tolist()

    # 将预测结果 写回task2_humor_rating文件中
    print('writing test result to task2_result')
    pd.DataFrame({'id': ids_test, 'text': text_test, 'task2_humor_rating': out_df['humor_rating']}).to_csv(
        result_path, index=False)


