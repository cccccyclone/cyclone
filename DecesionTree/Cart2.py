#-*- coding: utf-8 -*-
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
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
    for i in range(0,len(temp)-1):
        L_att.append('A'+str(flag))
        flag = int(flag)+1
    L_att.append('class')
    length = len(L_att)
    for i in range(0, length):
        dict[L_att[i]] = temp[i]
    data.append(dict)
    #以属性创建字典
    while 1:
        dict = {}
        line = file.readline().split()
        if len(line) == 0:
            break;
        #属性到值的映射
        for i in range(0 , length):
            dict[L_att[i]] = line[i]
        data.append(dict)
    #################################################
    return data,L_att[:]
    #################################################

#计算信息增益率,dattr为目标属性,attr为待测属性
def Gini(data ,  attr, dattr):
    #存储目标属性的计数
    Ent = {}
    #存储待测属性的计数
    Ent2 = {}
    g = float("inf")
    length =len(data)
    #目标属性集合及个数
    for single in data:
        if single[dattr] not in Ent:
            Ent[single[dattr]] = 0
        Ent[single[dattr]] += 1

    #计算待测属性的个数
    for single in data:
        if single[attr] not in Ent2:
            Ent2[single[attr]] = 0
        Ent2[single[attr]] += 1
    #print Ent2

    for k,v in Ent2.items():
        g2 = (float(v)/length)*Gini2(data,dattr,attr,k,1)+(float(length-v)/length)*Gini2(data,dattr,attr,k,0)
        #print attr,k,g2
        if g2 < g:
            g = g2
            gattr = k

    return g,gattr

def Gini2(data,dattr,attr,attrvalue,side):
    g = 1.0
    Ent = {}
    s = []
    for item in data:
        if side == 1:
            if item[attr] == attrvalue:
                s.append(item)
        else:
            if item[attr] != attrvalue:
                s.append(item)
    for single in s:
        if single[dattr] not in Ent:
                Ent[single[dattr]] = 0
        Ent[single[dattr]] += 1

    for key,value in Ent.items():
        g -= (float(value)/len(s))**2
    return g


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
    if len(data[0]) == 1:
        return Judge(Acount)
    #选择分裂属性
    selected,sattr = AttrSelect(data, attr, dattr)
    node = selected+':'+sattr
    Tree = {node:{}}

    single = [0,1]
    for v in single:
        sub = copy.deepcopy(attr)
        Tree[node][v] = CreateTree(split(data , selected ,sattr,v) , sub ,dattr)
    return Tree

#分割数据集
def split(data , selecetdattr ,value,side):
    res = []
    data2 = copy.deepcopy(data)
    for v in data2:
        if side == 1:
            if v[selecetdattr] == value:
                res.append(v)
        else:
            if v[selecetdattr] != value:
                res.append(v)

    #print res
    return res



#选择信息熵最高的属性,attr是属性的集合,dattr是目标属性
def AttrSelect(data , attr ,dattr):
    #print 'test:'
    #print data
    S = float("inf")
    result = copy.deepcopy(data)
    selected = 0
    #属性逐一计算增益率
    for elem in attr[:]:
        if elem == dattr:
            continue
        Sg, Ag = Gini(data,elem ,dattr )
        if Sg < S:
            S = Sg
            A = Ag
            selected = elem
    #print selected
    res = selected
    # 删除已分的属性
    attr.remove(selected)
    return res,A



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
#data, attr = GetInfo("Export_0")
data, attr= GetInfo('lenses')
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
''''''