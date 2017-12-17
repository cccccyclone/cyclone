# -*- coding:utf-8 -*-
#用于计算
import math
#用于打乱数组
import random
#用于深拷贝
import copy
from numpy import *
import numpy as np
import matplotlib.pyplot as plt



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


#计算欧式距离
def Distance(L1 , L2 ):
    s = 0.0

    s = pow( float(L1[0]) - float(L2[0]) , 2) + pow( float(L1[1]) - float(L2[1]) , 2)
    #返回欧氏距离
    return s

#随机选取中心点
def Get_Center(L , k):
    #打乱列表
    random.shuffle(L)
    CenterGroup = L[:k]
    #返回中心点集
    return CenterGroup

#生成簇,L1为中心点集,L2为非中心点集
def Cluster(center , data , k ):
    #以中心建簇
    sortedgroup = []
    for i in range(0,k):
        sortedgroup.append([])
    #以欧氏距离寻找最近中心点
    for elem in data:
        s = float("inf")
        for point in center:
            d = Distance(elem ,point)
            if d < s:
                s = d
                finalpoint = point
        #加入对应的中心点
        index = center.index(finalpoint)
        sortedgroup[index].append(elem)
    #返回簇
    return sortedgroup



#cener为中心点集，data为非中心点集,,k为簇数，max_iter为迭代次数
def Iteration(center,data,all_data,k,max_iter):
    times = 0
    while times <k:
        for i in range(0,k):
            newcenter = Avecount(data[i])
            center[i] = newcenter
        Cluster(center, all_data, k)
        times += 1
    print "迭代完成"
    return center,data

def Avecount(data):
    ave_x = 0.0
    ave_y = 0.0
    l = len(data)
    for item in data:
        ave_x += float(item[0])
        ave_y += float(item[1])
    return [ave_x/l,ave_y/l]

#c是颜色cValue = ['r','y','g','b','r','y','g','b','r']
#maker是形状
#s设置点大小

def Show(center,uncenter,pname):
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
    i = len(center)

    for  t in range(0,i):
        plt.plot(center[t][0], center[t][1], color[t] + 'v')
        for elem in uncenter[t]:
            plt.plot(elem[0], elem[1], color[t] + 'o')
    # 显示所画的图
    plt.show()

#用于给 gmm 调用
def Classfy2(filename,k,max_iter):
    data = GetInfo(filename)
    center = Get_Center(data, k)
    T = Cluster(center, data, k)
    center, uncenter = Iteration(center, T, data, k, max_iter)
    return center,uncenter
    #Show(center, uncenter,'K-Means')


def Classfy(filename,k,max_iter):
    data = GetInfo(filename)
    center = Get_Center(data, k)
    T = Cluster(center, data, k)
    center, uncenter = Iteration(center, T, data, k, max_iter)
    Show(center, uncenter,'K-Means')

#Classfy("iris",3,100)