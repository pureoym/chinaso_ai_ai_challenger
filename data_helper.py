#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : pureoym
# @Contact : pureoym@163.com
# @TIME    : 2018/10/10 9:40
# @File    : data_helper.py
# Copyright 2017 pureoym. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========================================================================
from tensorflow.python.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, Input, concatenate, Embedding
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.initializers import Constant
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.models import load_model
import pandas as pd
import jieba
import os
import numpy as np
import utils

# 数据保存地址
BASE_DIR = '/data0/search/ai_challenger/'
DATA_DIR = os.path.join(BASE_DIR, 'data/')
WORD2VEC = os.path.join(BASE_DIR, 'data/sgns.merge.bigram')
TRAIN_DATA_PATH = os.path.join(BASE_DIR, 'data/train/sentiment_analysis_trainingset.csv')
TEST_DATA_PATH = os.path.join(BASE_DIR, 'data/test/sentiment_analysis_testa.csv')
VALIDATION_DATA_PATH = os.path.join(BASE_DIR, 'data/train/sentiment_analysis_trainingset.csv')
WORD_INDEX = os.path.join(BASE_DIR, 'data/word_index.npy')
EMBEDDING_MATRIX = os.path.join(BASE_DIR, 'data/embedding_matrix.npy')
SEG_DATA = os.path.join(BASE_DIR, 'data/seg_data.csv')
PROCESSED_DATA = os.path.join(BASE_DIR, 'data/processed_data.csv')

MODEL_DIR = os.path.join(BASE_DIR, 'models/')
# NUMERIC_DATA = os.path.join(MODEL_DIR, 'numeric_data.csv')
# MODEL = os.path.join(MODEL_DIR, 'model.h5')

SEG_SPLITTER = ' '
word_index = {}

# Model Hyperparameters
EMBEDDING_DIM = 300  # 词向量维数
NUM_FILTERS = 100  # 滤波器数目
FILTER_SIZES = [2, 3, 4, 5]  # 卷积核
DROPOUT_RATE = 0.5
HIDDEN_DIMS = 64
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.2

# Training parameters
BATCH_SIZE = 64
NUM_EPOCHS = 1

# Prepossessing parameters
MAX_SEQUENCE_LENGTH = 300
MAX_NUM_WORDS = 150000  # 词典最大词数，若语料中含词数超过该数，则取前MAX_NUM_WORDS个
NUM_LABELS = 4  # 分类数目

# csv字段列表
columns = ['id', 'content', 'location_traffic_convenience',
           'location_distance_from_business_district', 'location_easy_to_find',
           'service_wait_time', 'service_waiters_attitude',
           'service_parking_convenience', 'service_serving_speed', 'price_level',
           'price_cost_effective', 'price_discount', 'environment_decoration',
           'environment_noise', 'environment_space', 'environment_cleaness',
           'dish_portion', 'dish_taste', 'dish_look', 'dish_recommendation',
           'others_overall_experience', 'others_willing_to_consume_again']

# 标签转换字典
label_dict = {'location_traffic_convenience': 'l1',
              'location_distance_from_business_district': 'l2',
              'location_easy_to_find': 'l3',
              'service_wait_time': 'l4',
              'service_waiters_attitude': 'l5',
              'service_parking_convenience': 'l6',
              'service_serving_speed': 'l7',
              'price_level': 'l8',
              'price_cost_effective': 'l9',
              'price_discount': 'l10',
              'environment_decoration': 'l11',
              'environment_noise': 'l12',
              'environment_space': 'l13',
              'environment_cleaness': 'l14',
              'dish_portion': 'l15',
              'dish_taste': 'l16',
              'dish_look': 'l17',
              'dish_recommendation': 'l18',
              'others_overall_experience': 'l19',
              'others_willing_to_consume_again': 'l20'}


def pre_process():
    """
    数据准备
    1 读取CSV
    2 分词，并保存结果
    3 获取词字典 word_index 按照词频倒排
    4 统计词频 按照顺序倒排
    5
    :return:
    """
    # 分词
    # 构建wordindex时将三个集合的文本都放进去
    data = pd.read_csv(TRAIN_DATA_PATH)
    data2 = pd.read_csv(TEST_DATA_PATH)
    data3 = pd.read_csv(VALIDATION_DATA_PATH)
    alldata = data.append(data2, ignore_index=True).append(data3, ignore_index=True)

    data['tokens'] = data['content'].map(utils.segment)
    alldata['tokens'] = alldata['content'].map(utils.segment)

    # 标签处理与统计
    # 将标签列名转换成['l1','l2',...,'l20']
    # 将[-2,-1,0,1]转换成[0,1,2,3]
    data.rename(columns=label_dict, inplace=True)
    for i in range(20):
        label_index = 'l' + str(i + 1)
        data[label_index] = data[label_index].map(lambda x: x + 2)
        series_i = pd.Series(data[label_index])
        print(label_index + ' value counts :\n')
        print(series_i.value_counts())

    # 获取word_index并保存
    allword_index = utils.get_word_index(alldata)
    np.save(WORD_INDEX, allword_index)

    # 序列化输入
    data['indexes'] = data['tokens'].map(utils.word2index)

    # 保存处理后的结果
    # data.to_csv(SEG_DATA)
    data.to_csv(PROCESSED_DATA)

    # 处理标签，生成每个模型需要的numeric_data，共计20个，并保存结果
    generate_numeric_data()
    print('generate_numeric_data')

    # 获取embeddings_index（加载预训练好的word2vec词典）
    embeddings_index = get_embeddings_index()
    print('get embeddings_index (or word2vec dict)')

    # 通过获取embeddings_index以及word_index，生成embedding_matrix
    embedding_matrix = generate_embedding_matrix(embeddings_index)
    np.save(EMBEDDING_MATRIX, embedding_matrix)
    print('generate_embedding_matrix and save to' + EMBEDDING_MATRIX)



def get_word_index(df):
    """
    统计语料分词词典，按照词频由大到小排序
    :param d0:
    :param d1:
    :return:
    """
    word_dict = {}
    for tokens in df['tokens']:
        words = tokens.split(SEG_SPLITTER)
        for word in words:
            if word in word_dict.keys():
                count = word_dict[word]
                word_dict[word] = count + 1
            else:
                word_dict[word] = 1
    word_dict = sorted(word_dict.items(), key=lambda x: x[1], reverse=True)
    for i, word in enumerate(word_dict):
        w = word[0]
        word_index[w] = i + 1
    return word_index


def get_embeddings_index():
    """
    加载预训练word2vec模型，返回字典embeddings_index
    :return: embeddings_index
    """
    embeddings_index = {}
    with open(WORD2VEC) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index


def generate_embedding_matrix(embeddings_index):
    """
    prepare embedding matrix
    使用embeddings_index，word_index生成预训练矩阵embedding_matrix。
    :param embeddings_index:
    :param word_index:
    :return:
    """
    num_words = min(MAX_NUM_WORDS, len(word_index) + 1)
    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= MAX_NUM_WORDS:
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix


def generate_numeric_data():
    """
    生成每个模型需要的numeric_data，共计20个
    :return:
    """
    # 获取预处理的数据
    processed_data = pd.read_csv(SEG_DATA, lineterminator='\n')

    # 处理标签
    for i in range(20):
        label_index = 'l' + str(i + 1)
        file_name = 'numeric_data_' + label_index + '.csv'
        # print(file_name)
        one_label_data = processed_data[['indexes', label_index]]
        one_label_data.rename(columns={label_index: 'labels'}, inplace=True)
        numeric_data = one_label_data[['indexes', 'labels']]
        numeric_data.to_csv(os.path.join(DATA_DIR, file_name), encoding='utf-8')
        print(pd.Series(numeric_data['labels']).value_counts())



if __name__ == '__main__':
    pre_process()