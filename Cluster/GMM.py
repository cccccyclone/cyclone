# -*- coding:utf-8 -*-
#用于计算
import math
#用于打乱数组
import random
#用于深拷贝
import copy
from numpy import *
import matplotlib.pyplot as plt
#导入k_means函数集
import K_means


#计算高斯密度
def Gauss(y,u,d):
    y = float(y)
    return (1.0/(sqrt(2*math.pi)*d))*exp(-pow(y-u,2)/(2*pow(d,2)))

#data为所有数据，alpha为模型分布概率,u,d为高斯密度参数，k为分类簇数
def Reaction(data,alpha,u,d,k):
    l = len(data)
    reaction_group = []
    for i in range(0,l):
        rg = []
        for j in range(0,k):
            temp = alpha[j]*Gauss(data[i][1],u[j],d[j])
            rg.append(temp)
        s = sum(rg)
        rg = [item/s for item in rg]
        reaction_group.append(rg)
    return reaction_group

#data数据集，rg为各点对模型的响应度，alpha为模型分布概率,u,d为高斯密度参数，k为分类簇数
def Updata_para(data,rg,alpha,u,d,k):
    l = len(data)
    u_new = []
    d_new = []
    alpha_new = []
    for  i in range(0,k):
        rjk = 0.0
        u_part = 0.0
        d_part = 0.0
        for j in range(0,l):
            rjk += rg[j][i]
            u_part += rg[j][i]*float(data[j][1])
            d_part += rg[j][i]*(float(data[j][1])-u[i])**2
        u_new.append(u_part)
        d_new.append(d_part)
        alpha_new.append(rjk)
    rjk_sum = sum(alpha_new)
    u_new = [item/rjk_sum for item in u_new]
    d_new = [item/rjk_sum for item in d_new]
    alpha_new =[item/l for item in alpha_new]



#计算获得初始化参数
def init_para(uncenter,data):
    all_count = len(data)
    l = len(uncenter)
    alpha = []
    u = []
    d = []
    for i in range(0,l):
        #初始化权值
        alpha.append(float(len(uncenter[i]))/all_count)
        #初始化均值
        u.append(mean([float(item[1]) for item  in uncenter[i]]))
        #初始化协方差
        d.append(varcount(array([float(item[1]) for item  in uncenter[i]])))
    return alpha,u,d

#计算方差
def varcount(arr):
    ave = mean(arr)
    sum = 0.0
    for item in arr:
        sum += (item-ave)**2
    return (1.0/len(arr))*sqrt(sum)

#迭代计算，filename为文件名，k为分类簇数，max_iter_kmeans为k-means分类迭代最大次数，max_iter_gmm为gmm迭代最大次数
def Iteration(filename,k,max_iter_kmeans,max_iter_gmm):
    data = K_means.GetInfo(filename)
    center, uncenter = K_means.Classfy(filename, k, max_iter_kmeans)
    alpha,u,d = init_para(uncenter,data)
    counter = 0
    while counter<max_iter_gmm:
        rg = Reaction(data, alpha, u, d, k)
        Updata_para(data, rg, alpha, u, d, k)
        counter += 1
    Show(rg,data,'GMM')


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
        k = rg[i].index(max(rg[i]))
        plt.plot(data[i][0], data[i][1], color[k] + 'o')

    # 显示所画的图
    plt.show()



Iteration("iris",3,100,50)
