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
    origin = []
    dataset = []
    label = []
    #打开文件
    with open(filename, "r") as csvfile:
        reader = csv.reader(csvfile)
        #逐条读取文件
        for line in reader:
            origin.append(line)
    origin = origin[1:]
    random.shuffle(origin)
    for item in origin:
        dataset.append(item[0])
        label.append(item[1])
    #存储取词后的邮件
    data = []
    for item in dataset:
        #直接把邮件取词处理
        data.append(textParse(item))
    return data,label

#初始数据预处理
def textParse(String):
    #保留字母
    listOfTokens = re.split(r'\W*', String)
    #字母小写化，去除纯数字，去除短字符串
    return [tok.lower() for tok in listOfTokens if (len(tok) > 2 and not tok.isdigit())]

#统计词条
def VocaProc(dataset,label):
    #邮件条数
    l = len(dataset)
    #print "The number of mails:",l
    #存储词条
    v_set = set([])
    #所有词条加入
    for item in dataset:
        v_set = v_set | set(item)
    #转化为list便于统计
    v_list = list(v_set)
    #统计词的种类数
    v_num = len(v_list)
    #print "The number of words:",v_num
    #所有转化后的邮件条目
    v_save = []
    for i in range(l):
        #单词条转化向量
        t = zeros(v_num)
        #统计每条邮件的每个词
        for item in dataset[i]:
            #若该词在词袋中
            if item in v_list:
                #每条邮件的转化
                t[v_list.index(item)] = 1
        v_save.append(t)
    v_save = np.array(v_save)
    return v_list,v_save

#计算p(w/c)
def ProbCount(v_save,label):
    #存储垃圾邮件/非垃圾邮件中各词的统计量
    c1_v_num = 0
    c0_v_num = 0
    #存储垃圾邮件/非垃圾邮件的统计量
    c1_num = 2
    c0_num = 2
    #邮件数
    l = len(v_save)
    #每个分类中的词统计
    v0_vec = ones(len(v_save[0]))
    v1_vec = ones(len(v_save[0]))
    for i in range(l):
        if int(label[i])== 0:
            v0_vec += v_save[i]
            c0_num += 1
        else:
            v1_vec += v_save[i]
            c1_num += 1
    c0_v_num += sum(v0_vec)
    c1_v_num += sum(v1_vec)
    #0类中的词统计概率
    p_v0_vec = np.log(v0_vec*1.0/c0_v_num)
    #1类中的词统计概率
    p_v1_vec = np.log(v1_vec*1.0/c1_v_num)
    p0_class = 1.0 * c0_num / (c0_num + c1_num)
    return p_v0_vec,p_v1_vec,p0_class

#对测试数据进行向量转换
def TestProc(test_data,v_list):
    test_vec = []
    for item in test_data:
        t = zeros(len(v_list))
        for elem in item:
            if elem in v_list:
                #不用统计个数？
                t[v_list.index(elem)] = 1
        test_vec.append(t)
    test_vec = np.array(test_vec)
    return test_vec


#计算属于各个类的概率
def classify(test_data,p_v0_vec,p_v1_vec,p0_class):
    p0 = sum(test_data*p_v0_vec)+log(p0_class)
    #print "p0_class：",p0_class
    p1 = sum(test_data * p_v1_vec) + log(1.0 - p0_class)
    if p0 >= p1:
        return 0
    else:
        return 1
#测试函数
def test(test_data,label,p_v0_vec,p_v1_vec,p0_class):
    l = len(test_data)
    count = 0
    for i in range(0,l):
        if int(label[i]) == classify(test_data[i],p_v0_vec,p_v1_vec,p0_class):
            count += 1
    print "rate:",count*1.0/l
    return count*1.0/l

def Main(filename,k):
    dataset, label = LoadTrainFile(filename)
    dataset = dataset[0:900]
    label = label[0:900]
    single = len(dataset)/k
    rate = 0.0
    print "训练数据：", single*(k-1)
    print "测试数据：", single
    #k折交叉验证
    for i in range(k):
        data_train = append(dataset[0:(i*single)],dataset[(i+1)*single:])
        label_train = append(label[0:(i*single)],label[(i+1)*single:])
        data_test = dataset[(i*single):(i+1)*single]
        label_test = label[(i*single):(i+1)*single]
        #获取词袋，转化后的向量，词袋总和
        v_list,v_save = VocaProc(data_train, label_train)
        #统计侮辱词条概率，分类概率
        p_v0_vec, p_v1_vec, p0_class = ProbCount(v_save, label_train)
        print "第" + str(i+1) + "折训练完"
        data_test = TestProc(data_test,v_list)
        print "*第" + str(i+1) + "折准确率："
        rate += test(data_test,label_test,p_v0_vec,p_v1_vec,p0_class)

    print "*平均准确率：",rate/k



Main("assignment1_data.csv",3)



