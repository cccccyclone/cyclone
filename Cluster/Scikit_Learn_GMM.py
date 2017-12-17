# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
import numpy as np

#数据导入
def GetInfo(filename):
    L = []
    temp = []
    for line in open(filename+".txt"):
        if line != "":
            temp = [float(item) for item in line.split()]
            L.append(temp)

        else:
            continue
    return L

#分类展示
def Show(data,flag,pname):
    r = len(data)
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    # 设置标题
    ax1.set_title(pname)
    # 设置X轴标签
    plt.xlabel('X')
    # 设置Y轴标签
    plt.ylabel('Y')
    color =['k','r','y','g','c','m','w']
    for i in range(0,r):
        plt.plot(data[i][0], data[i][1], color[flag[i]] + 'o')
    # 显示所画的图
    plt.show()

#分类函数
def Classfy(filename,k,max_it):
    L = GetInfo(filename)
    X = np.array(L)
    #调用混合高斯分布
    GMM = GaussianMixture(n_components=k,
                    covariance_type='full', tol = 0.001, reg_covar = 1e-06, max_iter = max_it, n_init = 1, init_params ='kmeans').fit(X)
    #获取分类后的标签
    labels = GMM.predict(X)
    #展示图形
    Show(X, labels,'scikit-learn-GMM')

#主函数
Classfy("iris",3,50)