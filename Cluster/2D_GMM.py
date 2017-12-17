# -*- coding:utf-8 -*-
#用于计算
import math
#用于打乱数组
import random
#用于深拷贝
import copy
import numpy as np
from numpy import *
import matplotlib.pyplot as plt
#导入k_means函数集
import K_means



#计算二维高斯
def D_Gauss(x,u,sigma):
    #计算维度
    s = len(x)
    g1 = 1.0/(2*math.pi*sqrt(trac(sigma)))
    # 1x2,2x2,2x1
    g2 = exp((-0.5)*multi((x-u),np.linalg.inv(sigma),(x-u).transpose()))
    return g1*g2


def multi(arr1,arr2,arr3):

    temp = np.dot(arr1,arr2)
    return np.dot(temp,arr3)


def trac(sigma):
    return sigma[0][0]*sigma[1][1]-sigma[0][1]*sigma[1][0]

#data为所有数据，alpha为模型分布概率,u,d为高斯密度参数，k为分类簇数
def Reaction(data,alpha,u,d,k):
    l = len(data)
    #保存响应度
    reaction_group = []
    #计算每个点的响应度
    for i in range(0,l):
        rg = []
        #对于每个簇
        for j in range(0,k):
            temp = alpha[j]*D_Gauss(data[i],u[j],d[j])
            rg.append(temp)
        rg = np.array(rg)
        s = sum(rg)
        _rg = rg/s
        reaction_group.append(_rg)
    return reaction_group

#data数据集，rg为各点对模型的响应度，alpha为模型分布概率,u,d为高斯密度参数，k为分类簇数
def Updata_para(data,rg,alpha,u,d,k):
    #数据点个数
    l = len(data)
    #存储新的均值、方差、权值
    u_new = []
    d_new = []
    alpha_new = []
    for  i in range(0,k):
        rjk = 0.0
        u_part = np.array([0.0]*2)
        d_part = np.array([[0.0,0.0],[0.0,0.0]])
        for j in range(0,l):
            rjk += rg[j][i]
            u_part += rg[j][i]*data[j]
            d_part += rg[j][i]*np.dot((data[j]-u[i]).transpose(),data[j]-u[i])
        u_new.append(u_part/rjk)
        d_new.append(d_part/rjk)
        alpha_new.append(rjk/l)




#计算获得初始化参数
def init_para(uncenter,data):
    #数据总数
    count = len(data)
    #分类簇数
    l = len(uncenter)
    #保存权值
    alpha = []
    #保存均值
    u = []
    #保存方差
    d = []
    #对于每个簇
    for i in range(0,l):
        alpha.append(round(float(len(uncenter[i]))/count,2))
        u.append([round(mean([float(item[0]) for item  in uncenter[i]]),2),round(mean([float(item[1]) for item  in uncenter[i]]),2)])
        d.append(cov(uncenter[i][:,0],uncenter[i][0:,1]))
    return np.array(alpha),np.array(u),d


#迭代计算，filename为文件名，k为分类簇数，max_iter_kmeans为k-means分类迭代最大次数，max_iter_gmm为gmm迭代最大次数
def Iteration(filename,k,max_iter_kmeans,max_iter_gmm):
    data = K_means.GetInfo(filename)
    data = np.array(data)
    center, uncenter = K_means.Classfy2(filename, k, max_iter_kmeans)
    ArrangeData(uncenter, k)
    alpha,u,d = init_para(uncenter,data)
    counter = 0
    while counter<max_iter_gmm:
        rg = Reaction(data, alpha, u, d, k)
        Updata_para(data, rg, alpha, u, d, k)
        counter += 1
    Show(rg,data,'GMM')


def ArrangeData(uncenter,k):
    for i in range(0,k):
        uncenter[i] = np.array(uncenter[i])


#c是颜色cValue = ['r','y','g','b','r','y','g','b','r']
#maker是形状
#s设置点大小

def Show(rg,data,pname):
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
    l = len(rg)
    for i in range(0,l):
        t = list(rg[i])
        k = t.index(max(t))
        plt.plot(data[i][0], data[i][1], color[k] + 'o')

    # 显示所画的图
    plt.show()

##
##a.transpose() # 转置
##numpy.linalg.inv(a) # 求逆
##
Iteration('iris',3,50,100)
