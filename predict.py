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
import os
import pandas as pd
from tensorflow.python.keras.models import load_model
from keras.models import load_model
import utils

BASE_DIR = '/data0/search/ai_challenger/'
TEST_DATA_PATH = os.path.join(BASE_DIR, 'data/test/sentiment_analysis_testa.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models/')
RESULT_PATH = os.path.join(BASE_DIR, 'result/result.csv')

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


def get_trained_models(model_path):
    """
    获取模型。
    :param model_path: 模型地址
    :return:20个模型的列表
    """
    models = {}
    for i in range(1,21):
        # model_path = os.path.join(MODEL_DIR, 'model_'+str(i)+'*')
        # model_name = 'model_' + str(i) + '_epoch_1.h5'
        model_name = 'model_1_epoch_1.h5'
        model_path = os.path.join(MODEL_DIR, model_name)load_model(model_path)
        models[i]=


def predict_and_generate_result(test_data, models, result_path):
    """
    根据模型与测试数据，预测结果。并将结果转换成所需的样式，并保存文件。
    :param test_data:
    :param models:
    :return:
    """
    pass


def get_result(input_indexes):
    rate = model.predict(np.matrix(input_indexes))
    l1=rate.tolist()[0]
    result = l1.index(max(l1)) -2
    return result





def test():
    df1['l1'] = df1['indexes'].map(get_result)
    df2 = df1[['content', 'indexes', 'l1']]
    df2.rename(columns={'l1':'location_traffic_convenience'}, inplace=True)



if __name__ == '__main__':
    # 处理原始测试数据数据，处理成模型输入所需要的形式。
    processed_test_data = data_preprocess(TEST_DATA_PATH)

    # 获取模型
    models = get_trained_models(MODEL_DIR)

    # 根据模型预测，将预测结果处理成提交数据
    predict_and_generate_result(processed_test_data, models, RESULT_PATH)

    l2 = l1.tolist()[0]
    max_index = l2.index(max(l2))
