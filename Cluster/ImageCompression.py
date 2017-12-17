# -*- coding:utf-8 -*-
#用于计算
import math
#用于打乱数组
import random
#用于深拷贝
import copy as cp
from numpy import *
import numpy as np
import matplotlib.pyplot as plt # plt 用于显示图片
from skimage import io

#处理像素数据
def GetInfo(filename):

    image = io.imread(filename)
    r = image.shape[0]
    c = image.shape[1]
    image = image.reshape(image.shape[0] * image.shape[1], 3)
    return image,r,c


#计算欧式距离
def Distance(L1 , L2 ):
    s = 0.0
    t = len(L1)
    p = L1 - L2
    for i in range(0,t):
        s += pow(float(p[i]) , 2)
    #返回欧氏距离
    return s

#随机选取中心点
def Get_Center(L , k):
    s = cp.deepcopy(L)
    #打乱列表
    random.shuffle(s)
    CenterGroup = s[:k]
    #返回中心点集
    return np.array(CenterGroup)

#生成簇,L1为中心点集,L2为非中心点集
def Cluster(center , data , k ):
    l = len(data)
    #记录簇分布
    record = np.array([0]*l)
    #记录中心点标记
    index = 0
    #以欧氏距离寻找最近中心点
    for i in range(0,l):
        s = float("inf")
        index = 0
        for p in range(0,k):
            d = Distance(data[i] ,center[p])
            if d < s:
                s = d
                index = p
        #加入对应的中心点
        record[i] = index
    #返回簇
    return record

#cener为中心点集，uncenter为非中心点集,,k为簇数，max_iter为迭代次数
def Iteration(center,record,data,k,max_iter):
    times = 0

    while times <max_iter:
        #print "第",times,"次"
        center = []
        for i in range(0,k):
            _temp = MeanCount(data,record,i)
            center.append([mean(_temp[:,0]),mean(_temp[:,1]),mean(_temp[:,2])])
            #print "new center:",newcenter
        record  = Cluster(center, data, k)
        times += 1
    print "迭代完成"
    return center,record

def MeanCount(data,record,index):
    l = len(data)
    r = []
    for i in range(0,l):
        if record[i]==index:
            r.append(data[i])
    return np.array(r)

def Modify(center,record,data,r,c):
    record = record.reshape(r, c)
    image = np.zeros((record.shape[0], record.shape[1], 3), dtype=np.uint8)
    for i in range(record.shape[0]):
        for j in range(record.shape[1]):
            image[i, j, :] = center[record[i, j]]
    return image

def Show(filename,compressed):
    image = io.imread(filename)
    ax1 = plt.subplot(211)
    ax1.set_title('origin')
    plt.imshow(image)
    ax2 = plt.subplot(212)
    ax2.set_title('compressed')
    plt.imshow(compressed)
    plt.show()

#filename为文件名，k为聚类数，max_iter为迭代次数
def Classfy(filename,k,max_iter):
    data, r, c = GetInfo(filename)
    center = Get_Center(data, k)
    record = Cluster(center, data, k)
    center, record = Iteration(center, record, data, k, 3)
    compressed = Modify(center, record, data, r, c)
    Show(filename, compressed)



#luffy.jpg聚类耗时较长，但效果明显

#panda.jpg耗时短，效果一般

Classfy('luffy.jpg',16,4)