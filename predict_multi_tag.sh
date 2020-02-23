#!/bin/bash
# ***********************************************************************
# **  功能描述：用于脚本启动BERT模型验证
# **  创建者： 微信公众号：数据时光者
# **  创建日期： 2020-02-22
# **  修改日期   修改人   修改内容
# ***********************************************************************

# 主目录
ROOT_PATH="./"

# 根据具体任务来划分文件夹
TASK_PATH=${ROOT_PATH}


# 数据存放目录
DATA_PATH=${TASK_PATH}'/data_input'
# BERT预训练模型目录
BERT_MODEL_NAME=${ROOT_PATH}'/bert_model'
# 训练集路径
TRAIN_DATA=${DATA_PATH}'/train.csv'
# 测试集路径
TEST_DATA=${DATA_PATH}'/test.csv'
# 测试集预测数据路径
TEST_PREDICT_DATA=${DATA_PATH}'/test_predict_online.csv'

# 手动选择导入哪个模型
MODEL_LOAD_PATH=${TASK_PATH}'/model_dump/xxxx'


python multi_tag_predict.py --bert_model_name=${BERT_MODEL_NAME} \
    --train_data=${TRAIN_DATA} \
    --test_data=${TEST_DATA} \
    --model_load_path=${MODEL_LOAD_PATH} \
    --test_predict_data=${TEST_PREDICT_DATA} 
