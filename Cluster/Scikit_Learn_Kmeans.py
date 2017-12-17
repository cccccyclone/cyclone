# -*- coding:utf-8 -*-


import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
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
def Show(data,flag,center,pname):
    x = []
    y = []
    fig = plt.figure()
    ax1 = fig.add_subplot(111)
    # 设置标题
    ax1.set_title(pname)
    # 设置X轴标签
    plt.xlabel('X')
    # 设置Y轴标签
    plt.ylabel('Y')

    color =['k','r','y','g','c','m','w']
    i = len(data)
    for t in range(0,i):
        print data[t]
        plt.plot(data[t][0], data[t][1], color[flag[t]] + 'o')
    for j in range(0,len(center)):
        plt.plot(center[j][0], center[j][1], color[j] + 'v')
    # 显示所画的图
    plt.show()

#分类函数
def Classfy(filename,k,max_it):
    L = GetInfo(filename)
    X = np.array(L)
    #调用Kmeans
    kmeans = KMeans(n_clusters=k, random_state=0, max_iter=max_it).fit(X)
    #获取标签
    labels = kmeans.labels_
    #获取中心点
    center = kmeans.cluster_centers_
    #展示图形
    Show(X, labels, center,'scikit-learn-kmeans')

#主函数
Classfy("iris",3,100)