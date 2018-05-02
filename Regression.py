#!/usr/bin/python3.6
# -*- coding: utf-8 -*-

# # =============神经网络用于回归=============

import numpy as np
from sklearn.neural_network import MLPRegressor  # 多层线性回归
from sklearn.preprocessing import StandardScaler
import ReadSamples as rs

dataMat = rs.readSamples()

X = dataMat[:1400, 10:13]
y = dataMat[:1400, 0:10]

testx = dataMat[1400:, 10:13]
testy = dataMat[1400:, 0:10]

scaler = StandardScaler()  # 标准化转换
scaler.fit(X)  # 训练标准化对象
X = scaler.transform(X)  # 转换数据集

scaler2 = StandardScaler()  # 标准化转换
scaler2.fit(testx)  # 训练标准化对象
testx = scaler2.transform(testx)  # 转换数据集

# solver='lbfgs',  MLP的求解方法：L-BFGS 在小数据上表现较好，Adam 较为鲁棒，SGD在参数调整较优时会有最佳表现（分类效果与迭代次数）；SGD标识随机梯度下降。
# alpha:L2的参数：MLP是可以支持正则化的，默认为L2，具体参数需要调整
# hidden_layer_sizes=(5, 6) hidden层2层,第一层5个神经元，第二层6个神经元)，2层隐藏层，也就有3层神经网络,4层节点
clf = MLPRegressor(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(100, 100,20), random_state=1)
clf.fit(X, y)
predicts=clf.predict(testx)
np.set_printoptions(suppress=True)  # 不采用科学计数法输出
for i in range(600):
    p=predicts[i]
    t=testy[i]
    print('预测结果：', predicts[i])  # 预测某个输入对象
    print('真实结果：', testy[i], '\n')
    for j in range(len(t)):
        print(p[j]-t[j])

cengindex = 0
for wi in clf.coefs_:
    cengindex += 1  # 表示底第几层神经网络。
    print('第%d层网络层:' % cengindex)
    print('权重矩阵维度:', wi.shape)
    print('系数矩阵：\n', wi)
