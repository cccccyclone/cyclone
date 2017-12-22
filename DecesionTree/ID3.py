
#-*- coding: utf-8 -*-
from numpy import *
import numpy as np
from math import log
import copy
import re
import random

############################################
#注意对序号等无关因素的处理，以免出错
############################################
#获取文件中的数据
def GetInfo(filename):
    data = []
    L_att = []
    temp = []
    flag = 1
    file = open(filename + ".txt",'r')
    #读取属性集合
    dict = {}
    #设定类名
    temp = file.readline().split()
    for i in range(0,4):
        L_att.append('A'+str(flag))
        flag = int(flag)+1
    L_att.append('class')
    length = len(L_att)
    for i in range(0, length):
        if len(temp)==6:
            if i==4:
                dict[L_att[i]] = temp[i]+"-"+temp[i+1]
                break
        dict[L_att[i]] = temp[i]
    data.append(dict)
    #以属性创建字典
    while 1:
        dict = {}
        line = file.readline().split()
        length = len(line)
        if len(line) == 0:
            break
        #属性到值的映射
        for i in range(0 , length):
            if len(line) == 6:
                if i == 4:
                    dict[L_att[i]] = line[i] + "-" + line[i + 1]
                    break
            dict[L_att[i]] = line[i]
        data.append(dict)
    #################################################
    return data,L_att[:]
    #################################################

#计算信息增益率,attr为目标属性,attr2为待测属性
def Entropy(data , attr, dattr ):
    #存储目标属性的计数
    Ent = {}
    #存储待测属性的计数
    Ent2 = {}
    #目标属性的熵
    Count = 0.0
    #待测属性的熵
    Count3 = 0.0
    #分区增益
    Count4 = 0.0
    length =len(data)
    #属性1集合及个数
    for single in data:
        if single[dattr] not in Ent:
            Ent[single[dattr]] = 0
        Ent[single[dattr]] += 1
    for k,v in Ent.items():
        Count -= (float)(func_log(v ,length ))
    #计算属性2的个数
    for single in data:
        if single[attr] not in Ent2:
            Ent2[single[attr]] = 0
        Ent2[single[attr]] += 1

    #单个属性的增益
    for k,v in Ent2.items():
        Count4 -= (float)(func_log(int(v), len(data)))
    #分母为0的情况
    if Count4 == 0.0:
        return 0

    #待测属性列
    for m,n in Ent2.items():
        Ent3 = {}
        Count2 = 0.0
        #数据列
        for sin2 in data:
            if sin2[attr] != m:
                continue
            #待测属性的目标属性列
            if sin2[dattr] not in Ent3:
                Ent3[sin2[dattr]] = 0
            Ent3[sin2[dattr]] += 1

        for k,v in Ent3.items():
            Count2 -= (float)(func_log(int(v) , n))
        Count3 += ((float)(n)/length)* Count2

    return float(Count - Count3)


#计算公式
def func_log(v ,length ):
    return ((float)(v)/length)*log((float)(v)/length ,2)

############################################################################################################################################
#生成决策树,data为数据集,attr为属性集
def CreateTree(datap , attr ,dattr):
    #深拷贝，防止被引用改变
    data = copy.deepcopy(datap)
    #目标属性值集合
    Acount = []
    #目标属性计数
    for v in data:
        Acount.append(v[dattr])

    #结果集仅有一种目标属性
    if Acount.count(Acount[0]) == len(Acount):
        return Acount[0]
    #结果集已被分完,因为还有列序号
    if len(data[0]) == 2:
        return Judge(Acount)
    #选择分裂属性
    selected = AttrSelect(data, attr, dattr)

    Tree = {selected:{}}

    values = []
    for v in data:
        values.append(v[selected.split('<=')[0]])
    #保留各项唯一的属性值
    single = list(set(values))
    single =sorted(single)
    for v in single:
        sub = copy.deepcopy(attr)
        Tree[selected][v] = CreateTree(split(data , selected ,v) , sub ,dattr)
    return Tree

#分割数据集
def split(data , selecetdattr ,value):
    res = []
    data2 = copy.deepcopy(data)
    for v in data2:
        if v[selecetdattr] == value:
            res.append(v)
    for v in res:
        del(v[selecetdattr])
    #print res
    return res

#属性二分化
def div(data , selecetdattr ,value):
    for elem in data:
        if value >= float(elem[selecetdattr]):
            elem[selecetdattr] = 0
        else:
            elem[selecetdattr] = 1



#选择信息熵最高的属性,attr是属性的集合,dattr是目标属性
def AttrSelect(data , attr ,dattr):
    S = 0.0
    result = copy.deepcopy(data)
    #属性逐一计算增益率
    for elem in attr[:]:
        if elem == dattr:
            continue
        if Entropy(result,elem ,dattr ) > S:
            S = Entropy(result ,elem ,dattr)
            selected = elem
    res = selected
    # 删除已分的属性
    attr.remove(selected)
    return res



#多数表决 待定
def Judge(data):
    t = 0
    Count = {}
    for v in data:
        if v not in Count:
            Count[v] = 0
        else:
            Count[v] += 1
    #返回计数最多的属性值
    return max(zip(Count.values(),Count.keys()))[1]




import TreeShow
#data为数据集,attr为属性集
data, attr= GetInfo("lenses")

print data
#保存原数据
data_s = copy.deepcopy(data)
#保存原数据
attr_s = copy.deepcopy(attr)
#建树
L = CreateTree(data_s, attr_s, 'class')
print L
#剪枝
#P = Pruning(L,data_s,data_test,attr_s,'label')
#画图
TreeShow.createPlot(L)
