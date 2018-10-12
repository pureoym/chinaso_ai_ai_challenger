#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : pureoym
# @Contact : pureoym@163.com
# @TIME    : 2018/10/9 14:49
# @File    : predict.py
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
# from keras.models import load_model
import pandas as pd
import os
import numpy as np
import textcnn_model
import utils

BASE_DIR = '/data0/search/ai_challenger/'
TEST_DATA_PATH = os.path.join(BASE_DIR, 'data/test/sentiment_analysis_testa.csv')
TEST_DATA_PATH2 = os.path.join(BASE_DIR, 'data/r1.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models/')
RESULT_PATH = os.path.join(BASE_DIR, 'result/result.csv')

SEG_SPLITTER = ' '
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 64
NUM_EPOCHS = 1

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


def data_preprocess(data_path):
    """
    数据预处理。将测试集文件处理成模型输入所需要的形式。并保存结果。
    :param data_path:测试集文件路径
    :return: 需要的dataframe
    """
    # 读取文件
    data = pd.read_csv(data_path)
    # 转换成词序号序列
    data['indexes'] = data['content'].map(utils.get_index_sequence_from_text)
    # 标签重命名
    data.rename(columns=label_dict, inplace=True)
    return data


# def get_trained_models(model_path):
#     """
#     获取模型。
#     :param model_path: 模型地址
#     :return:20个模型的列表
#     """
#     models = {}
#     for i in range(1, 21):
#         # model_path = os.path.join(MODEL_DIR, 'model_'+str(i)+'*')
#         # model_name = 'model_' + str(i) + '_epoch_1.h5'
#         model_name = 'model_1_epoch_1.h5'
#         model_path = os.path.join(MODEL_DIR, model_name)
#         models[i] = load_model(model_path)
#     return models


def train_model(input_path, epochs_number):
    """
    训练单个模型
    :param input_path:模型的输入
    :param epochs_number: 迭代次数
    :return:
    """
    m1 = textcnn_model.text_cnn_multi_class()
    # print(m1.summary())
    x_train, y_train, x_test, y_test = textcnn_model.pre_processing_multi_class(input_path)
    m1.fit(x_train, y_train,
           batch_size=BATCH_SIZE,
           epochs=epochs_number,
           validation_split=VALIDATION_SPLIT,
           shuffle=True)
    scores = m1.evaluate(x_test, y_test)
    print('test_loss: %f, accuracy: %f' % (scores[0], scores[1]))
    return m1


def predict_and_generate_result(test_data, models, result_path):
    """
    根据模型与测试数据，预测结果。并将结果转换成所需的样式，并保存文件。
    :param test_data:
    :param models:
    :return:
    """
    pass


def get_result(input_indexes):
    """
    通过indexes字段预测label字段
    需要将content字段经过预处理，处理成indexes字段。并需要预训练好的模型model。
    :param input_indexes:预处理好的保存在csv中的indexes字段
    :return:
    """
    rate = m1.predict(np.matrix(input_indexes))
    l1 = rate.tolist()[0]
    result = l1.index(max(l1)) - 2
    return result


def predict_and_save(df, result_path, label_name):
    """
    根据输入dateframe的indexes字段，通过模型预测label字段。并保存。
    :param df:输入dateframe，需要有indexes字段。
    :param result_path: 输出的文件路径
    :param label_name: 预测的标签名称
    :return:
    """
    df['prediction'] = df['indexes'].map(get_result)
    df2 = df[['content', 'prediction']]
    df2.rename(columns={'prediction': label_name}, inplace=True)
    df2.to_csv(result_path)
    return df2


# if __name__ == '__main__':
#     # 处理原始测试数据数据，处理成模型输入所需要的形式。
#     processed_test_data = data_preprocess(TEST_DATA_PATH)
#
#     # 获取模型
#     models = get_trained_models(MODEL_DIR)
#
#     # 根据模型预测，将预测结果处理成提交数据
#     predict_and_generate_result(processed_test_data, models, RESULT_PATH)
#
#     l2 = l1.tolist()[0]
#     max_index = l2.index(max(l2))


def get_training_path_and_result_path(index):
    training_file_name = 'data/numeric_data_l' + str(index) + '.csv'
    result_file_name = 'tmp/result_label_' + str(index) + '.csv'
    training_path = os.path.join(BASE_DIR, training_file_name)
    result_path = os.path.join(BASE_DIR, result_file_name)
    label_name = 'l' + str(index)
    return training_path, result_path, label_name


if __name__ == '__main__':
    # 切换工作路径
    import os
    os.chdir('/application/search/chinaso_ai_ai_challenger')

    # # 指定GPU
    # import os
    # import tensorflow as tf
    # import keras.backend.tensorflow_backend as KTF
    #
    # # 进行配置，每个GPU使用60%上限现存
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # 使用编号为1，2号的GPU
    # config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.9  # 每个GPU现存上届控制在90%以内
    # session = tf.Session(config=config)
    #
    # # 设置session
    # KTF.set_session(session)


    # 获取测试集
    test_df = pd.read_csv(TEST_DATA_PATH2)

    # 设置标签，并获取相关路径
    label_index = 2
    training_path, result_path, label_name = get_training_path_and_result_path(label_index)

    # 训练模型
    epochs_number = 1
    m1 = train_model(training_path, epochs_number)

    # 预测结果并保存，同时可以查看
    result_df = predict_and_save(test_df, result_path, label_name)
