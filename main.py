#!/usr/bin/env python
# coding: utf-8

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from PIL import ImageFile
import torchvision.models as models
import requests
from io import BytesIO
import sys
import argparse

# 颜色标签
colors_labels = ['#3D909E', '#3EA0A0', '#42A090', '#4983B3', '#498F76', '#508F65', '#6472B1', '#6A60AC', '#729156', '#7963AC', '#8C62AE', '#989246', '#9F8C4A', '#A3675F', '#A37C55', '#A56B5E', '#A5864F', '#A765B4', '#A85F5F', '#AB7C5E', '#AD609B', '#AE6187', '#AE775F', '#AF6179']
# 分类数量
n_classes = len(colors_labels)
# 把label转成对应的数字
class_to_num = dict(zip(colors_labels, range(n_classes)))
# 再转换回来，方便最后预测的时候使用
num_to_class = {v : k for k, v in class_to_num.items()}
# 设置使用CPU跑
device = 'cpu'

# resnet模型
def res_model(num_classes):
    model_ft = models.resnet50()
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes))
    return model_ft

def suggest(single_image_url):
    # 训练好的模型权重的文件路径
    model_path = './pre_res_model.ckpt'

    ## 初始化模型架构
    model = res_model(n_classes)

    # create model and load weights from checkpoint 加载模型权重
    model = model.to(device)
    model.load_state_dict(torch.load(model_path))

    # Make sure the model is in eval mode.
    # Some modules like Dropout or BatchNorm affect if the model is in training mode.
    model.eval()

    response = requests.get(single_image_url)
    response = response.content

    BytesIOObj = BytesIO()
    BytesIOObj.write(response)

    # 加载截断图片
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    # 读取图像文件
    img = Image.open(BytesIOObj)
    # 把PNG格式的图片像素数据转成RGB
    if img.mode != "RGB":
        img = img.convert("RGB")

    # 图像矩阵转换
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img = transform(img)
    # 3维转4维
    img = img.reshape(1, 3, 224, 224)

    # 模型预测
    with torch.no_grad():
        logits = model(img.to(device))

    # Take the class with greatest logit as prediction and record it.
    predictions = logits.argmax(dim=-1).cpu().numpy().tolist()

    # 打印结果
    return num_to_class[predictions[0]]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-url", dest="url", type=str)
    # 获取参数
    args = parser.parse_args()
    color = suggest(args.url)
    print(color)