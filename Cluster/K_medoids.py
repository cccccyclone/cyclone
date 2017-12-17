# -*- coding:utf-8 -*-
#用于计算
import math
#用于打乱数组
import random
#用于深拷贝
import copy as cp
from numpy import *
import matplotlib.pyplot as plt


#获取文件中的数据
def GetInfo(filename):
    L = []
    for line in open(filename+".txt"):
        if line != "":
            L.append(line.split())
        else:
            continue
    return L
#GetInfo("data")


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
    UncenterGroup = L[k:]
    #返回中心点集
    return CenterGroup,UncenterGroup

#生成簇,L1为中心点集,L2为非中心点集
def Cluster(center , uncenter , k ):
    #以中心建簇
    sortedgroup = []
    for i in range(0,k):
        sortedgroup.append([])
    #以欧氏距离寻找最近中心点
    for elem in uncenter:
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


#选取集合中到其他点代价最小的点
def CenterSelect(point,group):
    all_data = group
    all_data.append(point)
    l = len(all_data)
    sum = float("inf")
    for i in range(0,l):
        count = 0.0
        for j in range(0,l):
            if i != j:
                count += Distance(all_data[i],all_data[j])
        if (count < sum):
            res = all_data[i]
            sum = count
    return res



#cener为中心点集，data为非中心点集,,k为簇数，max_iter为迭代次数
def Iteration(center,uncenter,all_data,k,max_iter):
    times = 0
    while times <max_iter:
        group = cp.deepcopy(all_data)
        for i in range(0,k):
            #重置中心点
            center[i] = CenterSelect(center[i],uncenter[i])
            #从所有数据中移除中心点
            group.remove(center[i])
        #重置聚类
        uncenter = Cluster(center, group, k)
        times += 1
    print "迭代完成"
    return center,uncenter

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

def Classfy(filename,k,max_iter):
    data = GetInfo(filename)
    #划分中心点、非中心点集
    center ,uncenter= Get_Center(data, k)
    #初分类获得簇
    sortedgroup = Cluster(center,uncenter,  k)
    #迭代计算
    center, uncenter = Iteration(center, sortedgroup, data, k, max_iter)
    Show(center, uncenter,'K-medoids')

Classfy('iris',3,100)

