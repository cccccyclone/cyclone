# -*- coding:utf-8 -*-

from numpy import *
import numpy as np
import csv

#定义数据文件读取函数
def LoadTrainFile(filename):
    dataset = []
    label = []
    with open(filename, "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            dataset.append(line[1:])
            label.append(line[0])
    dataset = np.array(dataset[1:]).astype(np.float32)
    dataset = dataset/256
    label = np.array(label[1:]).astype(np.int32)
    return dataset,label

#定义数据文件读取函数
def LoadTestFile(filename):
    dataset = []
    with open(filename, "r", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            dataset.append(line)
    dataset = np.array(dataset[1:]).astype(np.float32)
    dataset = dataset/256
    csvfile.close()
    return dataset

#数据归一化
def Normalize(data):
    data_nomr = []
    for i in range(0,len(data)):
        max_value = np.max(data[i])
        min_value = np.min(data[i])
        if max_value == 0:
            data_nomr.append(data[i])
        else:
            temp = (data[i] - min_value)/(max_value - min_value)
            data_nomr.append(temp)
    return np.array(data_nomr)

#定义Sigmoid函数
def Sigmoid(vec):
    out = []
    for item in vec:
        out.append(1.0/(1 + math.exp(-item)))
    out = np.array(out)
    #print out
    return out


def BPnn(dataset,label,num_in,num_hide,num_out):
    #数据维度
    num_data = len(dataset[0])
    #初始化学习率
    rate_hide = 0.8
    rate_out = 0.8
    #初始化权重参数(784x15)
    w_hide =0.8*np.random.random((num_data, num_hide))-0.4
    #(15x10)
    w_out =0.4*np.random.random((num_hide, num_out))-0.2
    # 初始化偏置0,1x15
    offset_hide = np.zeros(num_hide)
    #1x10
    offset_out = np.zeros(num_out)
    #开始训练
    for i in range(0,len(dataset)):
        #设定标准输出值
        y = np.zeros(num_out)
        y[int(label[i])] = 1
        #前向传播,(1x784)x(784x15)=(1x15)
        inlayer_out = np.dot(dataset[i],w_hide) + offset_hide
        #(1x15)
        hidelayer_in = Sigmoid(inlayer_out)
        #(1x15)x(15x10)=(1x10)
        hidelayer_out = np.dot(hidelayer_in,w_out) + offset_out
        #(1x10)
        outlayer_in = Sigmoid(hidelayer_out)
        #反向传播
        #(1x10)x(1x10)
        gradient_out = ( y - outlayer_in) * outlayer_in * (1.0 - outlayer_in)
        #(1x15)x(1x15)x[(1x10)x(10x15)]
        gradient_hide = hidelayer_in * (1.0 - hidelayer_in) * np.dot( gradient_out,w_out.transpose())
        #更新隐藏层权重(1x15)
        for j in range(0, num_out):
            w_out[:, j] += rate_out * gradient_out[j] * hidelayer_in
        # 更新输入层权重
        for k in range(0, num_hide):
            w_hide[:, k] += rate_hide * gradient_hide[k] * dataset[i]
        # 阈值的更新
        offset_out += rate_out * gradient_out
        offset_hide += rate_hide * gradient_hide
    return w_hide, w_out, offset_hide, offset_out

#测试
def test(filename,datatest,w_hide, w_out, offset_hide, offset_out):
    with open(filename, "w",newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["ImageId", "Label"])

        #检验集数目
        num_test = len(datatest)
        #存储输出标签
        out_label = []
        #计数器
        count = 1
        #遍历每条数据集
        for i in range(0,num_test):
            #(1x784)x(784x15)=1x15
            inlayer_out = np.dot(datatest[i], w_hide) + offset_hide
            #print inlayer_out
            # (1x15)
            hidelayer_in = Sigmoid(inlayer_out)
            # (1x15)x(15x10)=(1x10)
            hidelayer_out = np.dot(hidelayer_in, w_out) + offset_out
            # (1x10)
            outlayer_in = Sigmoid(hidelayer_out)
            #最大值索引
            index = np.argmax(outlayer_in)
            t =[count,index]
            out_label.append(t)
            count +=1
        writer.writerows(out_label)
    return out_label


#导入数据集，对应标签
dataset,label = LoadTrainFile("train.csv")
datatest = LoadTestFile("test.csv")
num_in = 784
num_hide = 33
num_out = 10
w_hide,w_out,offset_hide,offset_out = BPnn(dataset,label,num_in,num_hide,num_out)

#print w_hide,w_out,offset_hide,offset_out
t = test('predict.csv',datatest,w_hide, w_out, offset_hide, offset_out)







