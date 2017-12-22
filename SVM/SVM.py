#!/usr/bin/python # -*- coding: utf-8 -*-
from numpy import *
import numpy as np

#加载数据集
def LoadDataSet(filename):
    data = []
    label = []
    fopen = open(filename)
    for line in fopen.readlines():
        temp = line.strip().split('\t')
        data.append([float(temp[0]), float(temp[1])])
        label.append([float(temp[2])])
    #加载数据
    data = np.array(data)
    #加载标签
    label = np.array(label)
    return data,label

#控制参数的上下限
def ModifyPara(des,H,L):
    if des > H:
        des = H
    if L > des:
        des = L
    return des

#计算误差值
def CalculateEk(t):
    #预测值
    all = float( np.dot((alpha*label).T , k[:,t]) + b)
    #减去标签
    Ek = all - float(label[t])
    return Ek

#选择步长最大的变量对
def selectJ(i, Ei):
    #使Ei差值生效
    ek[i] = [1,Ei]
    #非零值对应的下标列表
    validEcacheList = nonzero(ek[:,0])[0]
    if (len(validEcacheList)) > 1:
        #记录步长最大的一对参数
        maxindex = -1
        #记录步长
        maxDelta = 0
        #记录j差值
        Ej = 0
        #遍历每个有效差值
        for k in validEcacheList:
            #计算每个点的差值
            Ek = CalculateEk(k)
            #计算步长
            deltaE = abs(Ei - Ek)
            #若有更大的步长
            if (deltaE > maxDelta):
                #更新坐标
                maxindex = k
                #更新步长
                maxDelta = deltaE
                #更新参数对
                Ej = Ek
        return maxindex, Ej
    else:
        #若无有效点，则随机选取
        j = int(random.uniform(0,amount))
        #计算差值，并使之有效
        Ej = CalculateEk(j)
    return j, Ej

#计算参数eta的值
def EtaCount(i,j):
    return 2.0*k[i][j]-k[i][i]-k[j][j]
    #return  np.dot(2.0 * data[i,:],data[j,:].T) - np.dot(data[i,:],data[i,:].T) - np.dot(data[j,:],data[j,:].T)

#计算偏置值b
def BvalueCount(Ei,alphaIold,alphaJold,i,j):
    return b - Ei- multidot(label[i],(alpha[i]-alphaIold),k[i,j]) - multidot(label[j],alpha[j]-alphaJold,k[i,j])

#选择最佳参数对
def Alphachanged(i):
    global b
    #计算第一个点的差值
    Ei = CalculateEk(i)
    #a差值与标签的乘积，为正或为负
    value = label[i]*Ei
    #若需要优化
    if ((value < -tol) and (alpha[i] < C)) or ((value > tol) and (alpha[i] > 0)):
        #选择第二个参数点
        j,Ej = selectJ(i, Ei)
        #深拷贝以免数据改变造成影响
        alphaIold = alpha[i].copy()
        #深拷贝以免数据改变造成影响
        alphaJold = alpha[j].copy()
        #若方向相反
        if (label[i] != label[j]):
            #参数必须大于0
            L = max(0, alpha[j] - alpha[i])
            H = min(C, C + alpha[j] - alpha[i])
        else:
            # 参数必须大于0
            L = max(0, alpha[j] + alpha[i] - C)
            H = min(C, alpha[j] + alpha[i])
        #上下限相同时返回
        if L==H:
            #print "L==H"
            return 0
        #计算参数eta的大小
        eta = EtaCount(i,j)
        #要求i,j的修改量相同，修改方向相反
        if eta >= 0:
            #print "eta>=0"
            return 0
        #修改j变量
        alpha[j] -= np.dot(label[j],(Ei - Ej))/eta
        #控制修改的量
        alpha[j] = ModifyPara(alpha[j],H,L)
        #修正存储的误差变量
        Ek = CalculateEk(j)
        #使修改生效
        ek[j] = [1, Ek]
        #若修改的量太小
        if (abs(alpha[j] - alphaJold) < 0.00001):
            #print "j not moving enough"
            return 0
        #修改i变量
        alpha[i] += multidot(label[j],label[i],alphaJold - alpha[j])
        # 修正存储的误差变量
        Ek = CalculateEk(i)
        # 使修改生效
        ek[i] = [1, Ek]
        #计算偏置b1
        b1 = b - Ei- multidot(label[i],(alpha[i]-alphaIold),k[i,i]) - multidot(label[j],alpha[j]-alphaJold,k[i,j])
        # 计算偏置b2
        b2 = b - Ej- multidot(label[i],(alpha[i]-alphaIold),k[i,j]) - multidot(label[j],alpha[j]-alphaJold,k[j,j])
        if (alpha[i] > 0) and (alpha[i] < C):
            b = b1
        elif (alpha[j] > 0) and (alpha[j] < C):
            b = b2
        else:
            b = (b1 + b2)/2.0
        return 1
    else:
        return 0


#多矩阵相乘
def multidot(*kw):
    t = kw[0]
    for i in range(1,len(kw)):
        t = np.dot(t,kw[i])
    return t

#SMO算法
def SMO():
    #算法遍历次数
    itertimes = 0
    #是否遍历整个数据集
    iterate_entire = True
    # 参数对改变的次数
    alphaChangedTimes = 0
    while (itertimes < MaxIter) and ((alphaChangedTimes > 0) or (iterate_entire)):
        # 参数对改变的次数
        alphaChangedTimes = 0
        #遍历整个数据集
        if iterate_entire:
            #选择要改变的参数
            for i in range(amount):
                #记录改变次数
                alphaChangedTimes += Alphachanged(i)
                #print "fullSet, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
            #改变一对参数
            itertimes += 1
        #遍历非边界值
        else:
            #获取非边界值
            nonBound = nonzero((alpha> 0) * (alpha < C))[0]
            for i in nonBound:
                alphaChangedTimes += Alphachanged(i)
                #print "non-bound, iter: %d i:%d, pairs changed %d" % (iter,i,alphaPairsChanged)
            #改变一对参数
            itertimes += 1
        #遍历整个数据集完毕
        if iterate_entire:
            iterate_entire = False
        #若没有参数对改变，则再遍历整个数据集
        elif (alphaChangedTimes == 0):
            iterate_entire = True
        #print "iteration number: %d" % iter
    #返回
    return b,alpha

#http://blog.csdn.net/chenjianbo88/article/details/52373743
def InitWeight(num_in,num_out):
    r = 0.7*(num_out**(1./num_in))
    ##随机生成二维矩阵[-0.5,0.5]
    w = np.array([1,2,3])
    t = []
    for m in range(0,num_out):
        k = 0.0
        for n in range(0,num_in):
            k += w[n][m]**2
        t.append(k**(1./2))

    for i in range(0,num_in):
        for j in range(0,num_out):
            w[i][j] = r*(w[i][j]/t[j])

def DataTrans():
    temp = []
    for i in range(amount):
        location = []
        location.append(data[i][0]**2)
        location.append(data[i][1]**2)
        location.append(1.414*data[i][0]*data[i][1])
        temp.append(location)
    temp = np.array(temp)
    return temp

#计算权重w
def CalcWs():
    m,n = shape(data)
    w = np.zeros(n)
    for i in range(m):
        w += data[i,:].T*((alpha[i]*label[i]))
    return w

def f(x1,x2):
    m,n = shape(x1)
    x_sample = x1.flatten()
    y_sample = x2.flatten()
    l = len(x_sample)
    result = []
    for i in range(0,l):
        r = clf.predict([[x_sample[i],y_sample[i]]])
        result.append(r)
    r = np.array(result)
    print len(r)
    r = r.reshape(m,n)
    return r


import matplotlib.pyplot as plt
#显示图像分类后的结果
def Show():
    x0 = []
    y0 = []
    x1 = []
    y1 = []
    x2 = []
    y2 = []
    for i in range(len(data)):
        #选出正样本
        if int(label[i]) == 1:
            x1.append(data[i][0])
            y1.append(data[i][1])
        #选出负样本
        else:
            x2.append(data[i][0])
            y2.append(data[i][1])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    #正样本红色
    ax.scatter(x1,y1,s=30,c='red',marker='s')
    #负样本黑色
    ax.scatter(x2, y2, s=30, c='black', marker='s')
    # 数据数目
    n = 100
    # 定义x, y
    x = np.linspace(-7.5, 7.5, n)
    y = np.linspace(-7.5, 7.5, n)
    # 生成网格数据
    X, Y = np.meshgrid(x, y)
    # 绘制等高线
    C = plt.contour(X, Y, f(X, Y), 8, colors='black', linewidth=0.5)
    # 绘制等高线数据
    plt.clabel(C, inline=True, fontsize=10)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

def Kernel(datasample,delta):
    value = np.zeros(amount)
    for i in range(amount):
        diff = data[i,:] - datasample
        value[i] = np.dot(diff,diff.T)
    value = exp(value/(-1*delta**2))
    return value

def Kernel_2(delta):
    for i in range(amount):
        k[:,i] = Kernel(data[i,:],delta)
    return k

from mpl_toolkits.mplot3d import Axes3D
def Show_3D():
    global w
    x1 = []
    y1 = []
    z1 = []
    x2 = []
    y2 = []
    z2 = []
    for i in range(len(data)):
        if int(label[i]) == 1:
            x1.append(data[i][0])
            y1.append(data[i][1])
            z1.append(data[i][2])
        #选出负样本
        else:
            x2.append(data[i][0])
            y2.append(data[i][1])
            z2.append(data[i][2])
    ax = plt.subplot(111, projection='3d')
    ax.scatter(x1, y1, z1, c='y')
    ax.scatter(x2, y2, z2, c='r')
    X =arange(-20,20,1)
    Y = arange(-20,20,1)
    X, Y = np.meshgrid(X, Y)
    Z = ((-w[0])*X+(-w[1])*Y)/(w[2])
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='rainbow')
    ax.set_zlabel('Z')  # 坐标轴
    ax.set_ylabel('Y')
    ax.set_xlabel('X')

    plt.show()


from sklearn import svm
data,label = LoadDataSet("dataSet.txt")
X = data
y = ravel(label)
clf = svm.SVC()
clf.fit(X, y)

''''''
#数据量
amount,n = shape(data)
#data = DataTrans()

#设定alpha变量
alpha = np.array(zeros((amount,1)))
#偏置值
b = 0.0
#存储差值变量
ek =  np.array(zeros((amount,n)))
#
delta = 1000
#核函数变量
k = np.array(zeros((amount,amount)))
k = Kernel_2(delta)
#容错率
tol =0.001
#常数
C = 0.6
#迭代次数
MaxIter = 40
#获得偏置，拉格朗日参数
#b,alpha = SMO()

#输出图像
#data,label = LoadDataSet("dataSet_square.txt")
Show()
#Show_3D()


