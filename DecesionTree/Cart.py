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


############################################################################################################################################
#生成决策树,data为数据集,attr为属性
def CreateTree(datap , attr ,dattr):
    #深拷贝，防止被引用改变
    data = copy.deepcopy(datap)
    x_count = []
    #只有2个分界点了
    for item in data:
        if item[attr] not in x_count:
            x_count.append(item[attr])
    if(len(x_count) <= 2):
        return data
    #结果集已被分完
    if len(data) <=10:
        return data

    #选择点
    p = AttrSelect(data, attr, dattr)
    #节点名
    node = str(p)
    Tree = {node:{}}
    s = ['<=','>']
    for sub in s:
        Tree[node][sub] = CreateTree(split(data ,attr, sub ,p) , attr ,dattr)
    return Tree

#分割数据集
def split(data , selecetdattr ,side,value):
    res = []
    data2 = copy.deepcopy(data)
    for v in data2:
        if side == '<=':
            if float(v[selecetdattr]) <= float(value):
                res.append(v)
        else:
            if float(v[selecetdattr]) > float(value):
                res.append(v)
    return res





#选择信息熵最高的属性,attr是属性,dattr是目标属性
def AttrSelect(data , attr ,dattr):
    S = 0.0
    result = copy.deepcopy(data)
    #所有x集合
    unit = [example[attr] for example in result]
    #类型转化
    unit = map(float, unit)
    p =  cutpoint(unit,data,attr,dattr)
    return p

def cutpoint(point,data,attr,dattr):
    c1 = 0.0
    c2 = 0.0
    p = 0
    point = list(set(point))
    l = len(point)
    MinValue = float("inf")
    for i in range(0,l):
        left = []
        right = []
        for elem in data:
            #print float(elem[attr[0]]),float(point[i])
            if float(elem[attr]) <= float(point[i]):
                left.append(float(elem[dattr]))
            else:
                right.append(float(elem[dattr]))
        sum = Mindist(left,right)
        if sum <= MinValue:
            MinValue = sum
            p = point[i]
    return p


def Mindist(arr1,arr2):
    s1 = 0.0
    s2 = 0.0
    #求平均值
    c1 = float(summ(arr1))/len(arr1)
    for i1 in arr1:
        s1 += (c1 - float(i1)) ** 2
    if len(arr2) == 0:
        s2 = 0.0
    else:
        c2 = float(summ(arr2))/len(arr2)
        for i2 in arr2:
            s2 += (c2 - float(i2)) ** 2

    return s1+s2

def summ(arr):
    s = 0.0
    for i in range(0,len(arr)):
        s += float(arr[i])
    return s


#多数表决 待定
def Judge(data,dattr):
    t = 0.0
    if len(data) == 0:
        return 0
    for item in data:
        t += float(item[dattr])

    return round(float(t)/len(data),2)

#节点计数
def LeafCount(data):
    global count
    if type(data).__name__ != 'dict':
        count += 1
        return
    root = data.keys()[0]
    if type(data[root]).__name__=='dict':
        for item in data[root]:
            LeafCount(data[root][item])

#节点合并
def NodeMerge(data):
    global ls
    if type(data).__name__ != 'dict':
        ls += data
        return
    root = data.keys()[0]
    if type(data[root]).__name__ == 'dict':
        for item in data[root]:
            NodeMerge(data[root][item])

#某节点在训练集下的平方误差
def Deviation(L,tree,testdata,attr,dattr):
    cost = 0.0
    group = GroupSelect(testdata, L, tree,attr)
    for item in group:
        tree_save = L
        while 1:
            if (type(tree_save).__name__ != 'dict'):
                break
            node = tree_save.keys()[0]
            if float(item[attr]) <= float(node):
                tree_save  = tree_save[node]['<=']
            else:
                tree_save = tree_save[node]['>']

        #print 'lenth:',len(tree_save),Judge(tree_save,dattr),float(item[dattr])

        cost += round((Judge(tree_save,dattr) - float(item[dattr]))**2,2)
    return cost

#以t为单节点的树的平方误差
def Ct(L,tree,data_test,attr,dattr):
    global ls
    ls = []
    cost = 0.0
    NodeMerge(tree)

    ave = mean([float(item[dattr]) for item in ls])
    group = GroupSelect(data_test,L,tree,attr)
    for item in group:
        cost += (float(item[dattr])-ave)**2

    return cost

#选择符合条件的测试数据
def GroupSelect(data_test,tree_all,tree,attr):
    tag = tree.keys()[0]
    group = []
    for item in data_test:
        tree_save = tree_all
        while 1:
            if (type(tree_save).__name__ != 'dict'):
                break
            node = tree_save.keys()[0]
            if node == tag:
                group.append(item)
                break
            #print "type:",type(tree_save[node]).__name__
            if float(item[attr]) <= float(node):
                tree_save = tree_save[node]['<=']
            else:
                tree_save = tree_save[node]['>']
    return group

#剪枝并合并
def Cut(tree,tag,group):
    if (type(tree).__name__ != 'dict'):
        return
    node = tree.keys()[0]

    if float(tag) < float(node):
        if (type(tree).__name__ == 'dict') and tree[node]['<='].keys()[0] == tag:
            tree[node]['<='] = group
            return
        else:
            Cut(tree[node]['<='], tag, group)
    else:
        if (type(tree).__name__ == 'dict') and tree[node]['>'].keys()[0] == tag:
            tree[node]['>'] = group
            return
        else:
            Cut(tree[node]['>'], tag, group)

def Prunning(L,tree,data_test,attr, dattr):
    global ls
    global count
    global best_tree
    global a
    cost = float("inf")
    tree_group = []
    temp ={}
    while count != 2:
        a = float("inf")
        temp = CheckNodeCost(L, tree, data_test, attr, dattr)
        tag = temp.keys()[0]
        ls = []
        NodeMerge(temp)
        Cut(tree, tag, ls)

        count = 0
        LeafCount(tree)

        cost_tree = Deviation(L,tree,data_test,attr,dattr)
        if cost_tree < cost:
            cost = cost_tree
            best_tree = tree
            print best_tree


    return best_tree


#计算各内部节点的替换代价
def CheckNodeCost(L,tree,data_test,attr,dattr):
    global a
    global cut
    global count

    if type(tree).__name__ != 'dict':
        return
    root = tree.keys()[0]
    if type(tree[root]).__name__ == 'dict':
        for item in tree[root]:
            CheckNodeCost(L,tree[root][item],data_test,attr,dattr)
    #print 'inner node:', tree.keys()[0]

    #以t为单节点的树
    Ct_cost = Ct(L,tree,data_test,attr,dattr)

    #以t为根节点的树
    CT_cost = Deviation(L,tree,data_test,attr,dattr)

    #print "Ct_cost:",Ct_cost,"CT_cost:",CT_cost,"node:",tree.keys()[0]
    #print "*********************************"

    count = 0
    LeafCount(tree)
    g = float((Ct_cost - CT_cost)/(count - 1))
    if g < a:
        a = g
        cut = tree
    return cut


#树节点转换
def Cut_Tree_Show(tree,dattr):

    if type(tree).__name__ != 'dict':
        t = round(mean([float(item[dattr]) for item in tree]),2)
        return t
    root = tree.keys()[0]

    if type(tree[root]).__name__ == 'dict':
        for item in tree[root]:
            Cut_Tree_Show(tree[root][item],dattr)

best_tree = {}
count = 0
a = float("inf")
cut = {}
ls = []
cost = float("inf")
import TreeShow
#data为数据集,attr为属性集
data, attr = GetInfo("train")
#获得检验组
data_test, attr_test = GetInfo("test")
#保存原数据
data_s = copy.deepcopy(data)
#保存原数据
attr_s = copy.deepcopy(attr)
 #建树
L = CreateTree(data, 'A1', 'class')


k = Prunning(L,L,data_test,'A1', 'class')

#print count

''''''
#D = Deviation(L,data_test,'A1', 'class')
#print D
#print L
#C = Ct(L,data_test,'A1', 'class')
#print C
#LeafCount(L)
#TreeShow.createPlot(L)

#剪枝
#P = Pruning(L,data_s,data_test,attr_s,'label')
#画图

''''''
