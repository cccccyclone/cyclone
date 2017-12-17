#-*- coding: utf-8 -*-
from numpy import *
import numpy as np
from math import log
import copy
import re
import random
import csv

#定义数据文件读取函数
def LoadTrainFile(filename):
    dataset = []
    label = []
    origin = []
    with open(filename, "r") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            origin.append(line)
    #去除标签栏
    origin = origin[1:]
    for item in origin:
        label.append(item[0])
        dataset.append(item[1:])
    dataset = np.array(dataset).astype(np.int32)
    dataset = DataTrans(dataset)
    label = np.array(label).astype(np.int32)

    return dataset,label

#定义数据文件读取函数
def LoadTestFile(filename):
    dataset = []
    with open(filename, "r") as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            dataset.append(line)
    #去除标签栏
    dataset = dataset[1:]
    dataset = np.array(dataset).astype(np.int32)
    dataset = DataTrans(dataset)
    return dataset

def DataTrans(data):
    data_ = copy.deepcopy(data)
    f = lambda x:int(x!=0)
    for j in range(len(data_)):
        item = np.array([f(i) for i in data_[j]])
        data_[j] = item
    return data_

#计算p(w/c)
def ProbCount(v_save,label):
    #每条数据长度
    ls = len(v_save[0])
    #每个数字上的向量和
    c_num = ones(10)+1
    #数据数
    l = len(v_save)
    # 存储各数字的统计数
    p_cate = zeros(10)
    v_vec = ones((10,ls))
    #概率统计
    p_vec = zeros((10,ls))
    for i in range(l):
        v_vec[label[i]] += v_save[i]
        p_cate[label[i]] += 1

    for j in range(10):
        c_num[j] += sum(v_vec[j])
        p_vec[j] = np.log(v_vec[j]*1.0/c_num[j])
    print c_num

    #计算各类数字占比
    p_cate = p_cate*1.0/l

    return p_vec,p_cate


#计算属于各个类的概率
def classify(test_data,p_vec,p_cate):
    p = [0]*10
    for i in range(10):
        p[i] += sum(test_data*p_vec[i])+log(p_cate[i])
    index = p.index(max(p))
    return index

#测试函数
def test(test_data,p_vec,p_cate):
    with open("predict.csv", "w") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["ImageId", "Label"])
        # 存储输出标签
        out_label = []
        l = len(test_data)
        count = 1
        for i in range(l):
            index = classify(test_data[i],p_vec,p_cate)
            t = [count, index]
            out_label.append(t)
            count += 1
        writer.writerows(out_label)
    return out_label


''''''
def Main():
    data_train, label_train = LoadTrainFile("train.csv")
    data_test = LoadTestFile("test.csv")
    p_vec,p_cate = ProbCount(data_train,label_train)
    test(data_test,p_vec,p_cate)

Main()




