
#-*- coding: utf-8 -*-

import numpy as np
import math

def greedysearch(mapsize, blocks, init, goal):
    Start = init
    End = goal
    # 行数
    r = mapsize[0]
    # 列数
    c = mapsize[1]
    # 数据点阵
    DataSet = np.array([[0] * c] * r)
    blocks = blocks.tolist()
    blocks.extend([[item[0]+1,item[1]] for item in blocks])
    blocks.extend([[item[0], item[1]+1] for item in blocks])
    #print blocks
    # 不可达点
    CloseList = [item for item in blocks]
    OpenList = []
    # 各点的代价总和
    # 终点到当前点的距离
    H = np.array([[float("inf")] * c] * r)

    # 初始化
    OpenList.append(Start)
    H[Start[0], Start[1]] = ManDist(Start, End)
    Path = []
    while End not in OpenList:
        MinCostPoint = MinSelect(H, OpenList)
        Path.append(MinCostPoint)
        CloseList.append(MinCostPoint)
        OpenList = []
        NeighborCheck = NeighborCost2(MinCostPoint,OpenList,H,End,CloseList,DataSet)

    Path.append(End)
    return Path




def astarsearch(mapsize, blocks, init, goal):
    Start = init
    End = goal
    #行数
    r = mapsize[0]
    #列数
    c = mapsize[1]
    #数据点阵
    DataSet = np.array([[0] * c] *r)
    blocks = blocks.tolist()
    blocks.extend([[item[0] + 1, item[1]] for item in blocks])
    blocks.extend([[item[0], item[1] + 1] for item in blocks])
    #不可达点
    CloseList = [item for item in blocks]
    OpenList = []
    #各点的代价总和
    CostValue = np.array([[float("inf")] * c] *r)
    #起点到当前点的距离
    G = np.array([[float("inf")] * c] *r)
    #终点到当前点的距离
    H = np.array([[float("inf")] * c] * r)
    #各点的父节点
    Parent = np.array([[[0, 0]] * c] * r)

    #初始化
    OpenList.append(Start)
    CostValue[Start[0],Start[1]] = 0
    G[Start[0],Start[1]] = 0
    H[Start[0],Start[1]] = ManDist(Start, End)

    while End not in OpenList:
        MinCostPoint = MinSelect(CostValue, OpenList)

        CloseList.append(MinCostPoint)
        OpenList.remove(MinCostPoint)

        NeighborCheck = NeighborCost(MinCostPoint,OpenList,Parent,H,G,End,CostValue,CloseList,DataSet)
    Path = []
    GetPath(Parent,End,Start,Path)
    Path.append(End)
    return Path

#计算两点间曼哈顿距离
def ManDist(point_1,point_2):

    return abs(point_1[0] - point_2[0]) + abs(point_1[1] - point_2[1])

#找到代价最小的坐标
def MinSelect(dataset,openlist):

    val = float("inf")
    loc = [0,0]
    for item in openlist:
        datavalue = dataset[item[0],item[1]]
        if datavalue <= val:
            val = datavalue
            loc = item
    return loc

#计算各邻居节点
def NeighborCost(point,openlist,parent,h,g,end,costvalue,closelist,dataset):
    r = point[0]
    c = point[1]
    ReachAble(openlist, parent, point, h, g, end, costvalue, closelist, dataset, r - 1, c + 1)
    ReachAble(openlist, parent, point, h, g, end, costvalue, closelist, dataset, r, c + 1)
    ReachAble(openlist, parent, point, h, g, end, costvalue, closelist, dataset, r + 1, c + 1)
    ReachAble(openlist, parent, point, h, g, end, costvalue, closelist, dataset, r + 1, c)
    ReachAble(openlist, parent, point, h, g, end, costvalue, closelist, dataset, r - 1, c)
    ReachAble(openlist,parent,point,h,g,end,costvalue,closelist,dataset,r+1, c- 1)
    ReachAble(openlist,parent,point,h,g,end,costvalue,closelist,dataset,r, c - 1)
    ReachAble(openlist,parent,point,h,g,end,costvalue,closelist,dataset,r-1,c-1)

    #print openlist

#判断是否可达
def ReachAble(openlist,parent,point,h,g,end,costvalue,closelist,dataset,r,c):
    #总行数
    rboder = len(dataset)
    #总列数
    cboder = len(dataset[0])
    #点越界处理
    if r<0 or c<0 or r>=rboder or c>=cboder:
        return
    res = [r,c]
    #已选择过该点
    if res in closelist:
        return

    if res not in openlist :
        #加入可选列表
        openlist.append(res)
        #print "openlist:"
        #print openlist
        SetParent(res, point, parent)
        CountValue(res, point,costvalue,g,h,end)
        return
    #若该点已在可选列表中
    if  res in openlist :
        t = [res[0] - point[0],res[1] - point[1]]

        #该点到起点的距离
        dist_g = round(math.sqrt(t[0]**2 + t[1]**2),1) + g[point[0]][point[1]]
        #列表中方案的到起点的距离
        parent_g = g[res[0]][res[1]]
        #如果当前状况比方案好，则替代
        if dist_g <  parent_g:
            SetParent(res, point, parent)
            g[res[0]][res[1]] = dist_g
            h[res[0]][ res[1]] = ManDist(end,res)
            costvalue[res[0]][res[1]] = dist_g + ManDist(end,res)


#为该点设置前置节点
def SetParent(child_point,parent_point,Parent):
    Parent[child_point[0],child_point[1]] = parent_point

#计算移动代价
def CountValue(child_point,parent_point,CostValue,G,H,End):
    #计算到终点的距离
    dist_h =  ManDist(child_point,End)
    t = [child_point[0] - parent_point[0],child_point[1] - parent_point[1]]
    #计算起点节点的距离（从父节点到起点的距离+该节点到父节点的距离）
    dish_g = round(math.sqrt(t[0]**2 + t[1]**2),1)+ G[parent_point[0],parent_point[1]]
    #保存到起点的距离
    G[child_point[0],child_point[1]] = dish_g
    # 保存到终点的距离
    H[child_point[0],child_point[1]] = dist_h
    #保存总代价
    CostValue[child_point[0],child_point[1]] = dish_g + dist_h

#回溯得到路径
def GetPath(Parent,End,Start,Path):
    p = (Parent[End[0],End[1]]).tolist()

    if p != Start:
        GetPath(Parent,p,Start,Path)
    Path.append(p)


########################################################################################################################
#计算各邻居节点
def NeighborCost2(point,openlist,h,end,closelist,dataset):
    r = point[0]
    c = point[1]
    ReachAble2(openlist,  point, h, end, closelist, dataset, r - 1, c + 1)
    ReachAble2(openlist,  point, h, end, closelist, dataset, r, c + 1)
    ReachAble2(openlist,  point, h, end, closelist, dataset, r + 1, c + 1)
    ReachAble2(openlist,  point, h, end, closelist, dataset, r + 1, c)
    ReachAble2(openlist,  point, h, end, closelist, dataset, r - 1, c)
    ReachAble2(openlist,  point, h, end, closelist, dataset, r + 1, c - 1)
    ReachAble2(openlist,  point, h, end, closelist, dataset, r, c - 1)
    ReachAble2(openlist,  point, h, end, closelist, dataset, r - 1, c - 1)
    #print openlist

#计算是否可达
def ReachAble2(openlist,point,h,end,closelist,dataset,r,c):
    #总行数
    rboder = len(dataset)
    #总列数
    cboder = len(dataset[0])
    #点越界处理
    if r<=0 or c<=0 or r>=rboder or c>=cboder:
        return
    res = [r,c]
    #已选择过该点
    if res in closelist:
        return
    #未选择的邻居节点都纳入考虑范围
    openlist.append(res)
    #计算距离
    CountValue2(res, h,end)
    return


#距离结算
def CountValue2(child_point,H,End):
    #计算到终点的距离
    dist_h =  ManDist(child_point,End)
    # 保存到终点的距离
    H[child_point[0],child_point[1]] = dist_h




