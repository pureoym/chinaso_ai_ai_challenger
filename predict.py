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

BASE_DIR = '/data0/search/ai_challenger/'
TEST_DATA_PATH = os.path.join(BASE_DIR, 'data/test/sentiment_analysis_testa.csv')
MODEL_DIR = os.path.join(BASE_DIR, 'models/')
RESULT_PATH = os.path.join(BASE_DIR, 'result/result.csv')


def data_preprocess(data_path):
    """
    数据预处理。将测试集文件处理成模型输入所需要的形式。并保存结果。
    :param data_path:测试集文件路径
    :return: 需要的dataframe
    """
    data = pd.read_csv(data_path)
    data['tokens'] = data['content'].map(segment)
    return output_data


def get_trained_models(model_path):
    """
    获取模型。
    :param model_path: 模型地址
    :return:20个模型的列表
    """
    pass


def predict_and_generate_result(test_data, models, result_path):
    """
    根据模型与测试数据，预测结果。并将结果转换成所需的样式，并保存文件。
    :param test_data:
    :param models:
    :return:
    """
    pass


if __name__ == '__main__':
    # 处理原始测试数据数据，处理成模型输入所需要的形式。
    processed_test_data = data_preprocess(TEST_DATA_PATH)

    # 获取模型
    models = get_trained_models(MODEL_DIR)

    # 根据模型预测，将预测结果处理成提交数据
    predict_and_generate_result(processed_test_data, models, RESULT_PATH)
