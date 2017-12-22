#!/usr/bin/python # -*- coding: utf-8 -*-
from numpy import *
import numpy as np

def CircleGenerator():
    data = []
    label = []
    samples_num = 150
    t = np.random.random(size=samples_num) * 2 * np.pi - np.pi
    x = np.cos(t)
    y = np.sin(t)
    i_set = np.arange(0,samples_num,1)
    for i in i_set:
        len = np.sqrt(np.random.random())
        x[i] = x[i] * len
        y[i] = y[i] * len

    f = open('dataSet.txt', 'w')
    for i in i_set:
        temp = []
        t = x[i]**2+y[i]**2
        if t>0.36 and t<0.64:
            continue
        if t<=0.36 :
            flag =-1
        else:
            flag =1
        f.writelines(str(round(x[i]*5,3))+"\t"+str(round(y[i]*5,3))+"\t"+str(flag)+'\n')
        temp.append(round(x[i] * 5, 3))
        temp.append(round(y[i] * 5, 3))
        data.append(temp)
        label.append(flag)
    data = np.array(data)
    label = np.array(label)
    f.close()
    return data,label

def SquareGenerator():
    f = open('dataSet_square.txt', 'w')
    x1,y1 = (np.random.rand(2,20)+0.1)*10
    x2, y2 = (np.random.rand(2, 20)+0.1) * 10
    x2 = -x2
    x3, y3 = (np.random.rand(2, 20)+0.1) * (-10)
    x4, y4 = (np.random.rand(2, 20)+0.1) * 10
    y4 = -y4
    '''
    for i in range(0,10):
        f.writelines(str(round(x1[i], 3)) + "\t" + str(round(y1[i], 3)) + "\t" + str(1) + '\n')
        f.writelines(str(round(x2[i], 3)) + "\t" + str(round(y2[i], 3)) + "\t" + str(-1) + '\n')
        f.writelines(str(round(x3[i], 3)) + "\t" + str(round(y3[i], 3)) + "\t" + str(1) + '\n')
        f.writelines(str(round(x4[i], 3)) + "\t" + str(round(y4[i], 3)) + "\t" + str(-1) + '\n')
        '''
    f.close()
    ShowSquare(x1, x2, x3, x4, y1, y2, y3, y4)
    return 0

import matplotlib.pyplot as plt
def ShowSquare(x1,x2,x3,x4,y1,y2,y3,y4):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(x1, y1, s=30, c='red', marker='s')
    ax.scatter(x2, y2, s=30, c='black', marker='s')
    ax.scatter(x3, y3, s=30, c='red', marker='s')
    ax.scatter(x4, y4, s=30, c='black', marker='s')
    plt.grid(True)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


def ShowCircle(data,label):
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
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

import SVM
#SquareGenerator()
#data,label = CircleGenerator()
data,label = SVM.LoadDataSet("dataSet.txt")
ShowCircle(data,label)





'''    
    plt.figure(figsize=(10,10.1),dpi=125)
    plt.plot(x,y,'ro')
    _t = np.arange(0,7,0.1)
    _x = np.cos(_t)
    _y = np.sin(_t)
    plt.plot(_x,_y,'g-')
    plt.xlim(-1.1,1.1)
    plt.ylim(-1.1,1.1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Random Scatter')
    plt.grid(True)
    #plt.savefig('imag.png')
    plt.show()
'''