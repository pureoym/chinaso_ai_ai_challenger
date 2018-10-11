#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : pureoym
# @Contact : pureoym@163.com
# @TIME    : 2018/10/8 11:21
# @File    : model.py
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

# transformer 模型
# 带attention的lstm

from tensorflow.python.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, Input, concatenate, Embedding
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.initializers import Constant
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.models import load_model




from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Dropout, Input, concatenate, Embedding
from keras.models import Model
from keras.initializers import Constant
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import load_model

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


def pre_processing_multi_class(path):
    """
    预处理。获取训练集，测试集。
    :return:
    """

    # 获取数字化的数据集
    d1 = pd.read_csv(path)
    d1['index_array'] = d1['indexes'].map(lambda x: x.split(SEG_SPLITTER))
    sequences = d1['index_array']
    data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    labels = d1['labels'].values.reshape(-1, 1)
    labels = to_categorical(labels)
    # print('Shape of data tensor:', data.shape)
    # print('Shape of label tensor:', labels.shape)

    # 切分训练集和测试集
    data_size = data.shape[0]
    indices = np.arange(data_size)
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    train_test_samples = int(TEST_SPLIT * data_size)

    x_train = data[:-train_test_samples]
    y_train = labels[:-train_test_samples]
    x_test = data[-train_test_samples:]
    y_test = labels[-train_test_samples:]
    # print('Shape of data x_train:', x_train.shape)
    # print('Shape of label y_train:', y_train.shape)
    # print('Shape of data x_test:', x_test.shape)
    # print('Shape of label y_test:', y_test.shape)
    return x_train, y_train, x_test, y_test


def text_cnn_multi_class():
    """
    构建多分类text_cnn模型
    :return:
    """
    # 输入层
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')

    # 嵌入层
    # load pre-trained word embeddings into an Embedding layer
    # note that we set trainable = False so as to keep the embeddings fixed
    embedding_matrix = np.load(EMBEDDING_MATRIX)
    num_words = embedding_matrix.shape[0] + 1
    embedding_layer = Embedding(num_words,
                                EMBEDDING_DIM,
                                embeddings_initializer=Constant(embedding_matrix),
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False)
    embedded_sequences = embedding_layer(sequence_input)

    # 卷积层
    convs = []
    for filter_size in FILTER_SIZES:
        l_conv = Conv1D(filters=NUM_FILTERS,
                        kernel_size=filter_size,
                        activation='relu')(embedded_sequences)
        l_pool = MaxPooling1D(MAX_SEQUENCE_LENGTH - filter_size + 1)(l_conv)
        l_pool = Flatten()(l_pool)
        convs.append(l_pool)
    merge = concatenate(convs, axis=1)

    # 全连接层
    x = Dropout(DROPOUT_RATE)(merge)
    x = Dense(HIDDEN_DIMS, activation='relu')(x)

    preds = Dense(units=NUM_LABELS, activation='softmax')(x)

    model = Model(sequence_input, preds)
    model.compile(loss="categorical_crossentropy",
                  optimizer="rmsprop",
                  metrics=['acc'])

    return model


def train_and_save_model(model_index_list, epochs_number, model):
    """
    训练并保存模型
    :param model_index_list:
    :param epochs_number:
    :param model:
    :return:
    """
    for i in model_index_list:
        print('################[ multi_epochs_model_' + str(i) + ' ]################')
        path = os.path.join(DATA_DIR, 'numeric_data_l' + str(i) + '.csv')
        x_train, y_train, x_test, y_test = pre_processing_multi_class(path)
        model.fit(x_train, y_train,
                  batch_size=BATCH_SIZE,
                  epochs=epochs_number, validation_split=VALIDATION_SPLIT,
                  shuffle=True)
        scores = model.evaluate(x_test, y_test)
        print('test_loss: %f, accuracy: %f' % (scores[0], scores[1]))
        model.save(os.path.join(MODEL_DIR, 'model_' + str(i) + '_epoch_' + str(epochs_number) + '.h5'))


###########################################
## 贴到此分隔符以上

if __name__ == '__main__':
    # 根据原始数据生成训练用数据，如果已经处理好，则直接调用结果
    # prepare_data()

    # 第一步，在jupyter里构建模型
    # 构建模型
    model = text_cnn_multi_class()
    model.summary()

    # 第二部，训练并保存模型
    # 训练20个模型
    # epochs_number_1 = 1
    # model_epoch_1 = range(1, 3)
    # train_and_save_model(model_epoch_1, epochs_number_1, model)

    # epochs_number_1 = 1
    # model_epoch_1 = range(1, 21)
    # train_and_save_model(model_epoch_1, epochs_number_1, model)

    # epochs_number_3 = 3
    # model_epoch_3 = [5, 8, 10, 11, 12, 13, 14, 15, 16, 17, 19, 20]
    # train_and_save_model(model_epoch_3, epochs_number_3, model)

    # epochs_number_10 = 10
    # model_epoch_10 = [8, 13, 15, 16, 17, 19, 20]
    # train_and_save_model(model_epoch_10, epochs_number_10, model)

    train_and_save_model([16], 10, model)
