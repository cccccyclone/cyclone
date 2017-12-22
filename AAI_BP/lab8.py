import numpy as np
import math
import  cifar10

num_in = 3072
num_hide = 70
num_out = 10

# 初始化学习率
rate_hide = 0.1
rate_out = 0.1
# 初始化权重参数，控制在[-0.5,0.5]
w_hide = np.random.random((num_in, num_hide))-0.5
# (56x10)
w_out = np.random.random((num_hide, num_out))-0.5
# 初始化偏置
offset_hide = np.zeros(num_hide)
# 1x10
offset_out = np.zeros(num_out)

#定义Sigmoid函数
def Sigmoid(vec):
    out = []
    for item in vec:
        out.append(1.0/(1 + math.exp(-item)))
    out = np.array(out)
    return out

def InitWeight(n_in,n_out):
    r = 0.7*(n_out**(1./n_in))
    #随机乘车二维矩阵[-0.5,0.5]
    w = np.random.random((n_in, n_out))-0.5



def BPnn(data,one_hot_labels):

    #数据整合成一维
    dataset = data_merge(data)
    #dataset = TransGrey(dataset)
    #数据量
    num_data = len(dataset)
    #开始训练
    for i in range(0,num_data):
        if i%1000 == 0:
            print(str(i)+"times")
        #设定标准输出值
        y = one_hot_labels[i]
        #前向传播(1x3072)x(3072xH)+(1xH)
        inlayer_out = np.dot(dataset[i],w_hide) + offset_hide
        #激活函数
        hidelayer_in = Sigmoid(inlayer_out)
        #前向传播隐藏层(1xH)x(Hx10)+(1x10)
        hidelayer_out = np.dot(hidelayer_in,w_out) + offset_out
        #激活函数
        outlayer_in = Sigmoid(hidelayer_out)
        #后向传播修正参数
        BackPro(dataset,i,y,hidelayer_in,outlayer_in)
    return w_hide, w_out, offset_hide, offset_out

def BackPro(dataset,i,y,hidelayer_in,outlayer_in):
    global offset_out
    global offset_hide
    # 反向传播计算梯度
    gradient_out = (y - outlayer_in) * outlayer_in * (1.0 - outlayer_in)
    # 计算梯度
    gradient_hide = hidelayer_in * (1.0 - hidelayer_in) * np.dot(gradient_out, w_out.transpose())
    # 更新隐藏层权重
    for j in range(0, num_out):
        w_out[:, j] += rate_out * gradient_out[j] * hidelayer_in
    # 更新输入层权重
    for k in range(0, num_hide):
        w_hide[:, k] += rate_hide * gradient_hide[k] * dataset[i]
    # 阈值的更新
    offset_out += rate_out * gradient_out
    offset_hide += rate_hide * gradient_hide

#开始训练
def train(images, one_hot_labels):
    BPnn(images, one_hot_labels)
    return 0


#训练集合成一维
##这步很关键，之前用循环遍历的方式合成数组，导致花费大量的时间，还以为是代码死循环了
##flatten效率非常高，很好用
def data_merge(images):
    l = len(images)
    all = []
    for i in range(0,l):
        item = images[i].flatten()
        all.append(item)
    all = np.array(all)
    print("merged ok~")
    return all
'''
#分解训练集
def data_split(images,img_size):
    l = len(images)
    all = []
    for i in range(0,l):
        temp = images[i].reshape(32, 32, -1)
        all.append(temp)
    all = np.array(all)
    return all
'''
def TransGrey(datasample):
    l = len(datasample)
    r = []
    for i in range(l):
        v = []
        for j in range(1024):
            g = 0.299*datasample[i][j*3]+0.587*datasample[i][j*3+1]+0.587*datasample[i][j*3+2]
            v.append(g)
        r.append(v)
    print("grayed ok~")
    return np.array(r)


def predict(images):
    # Return a Numpy ndarray with the length of len(images).
    # e.g. np.zeros((len(images),), dtype=int) means all predictions are 'airplane's
    datatest = data_merge(images)
    l = len(datatest)
    out_label = []
    for i in range(0, l):
        #前向过输入层
        inlayer_out = np.dot(datatest[i], w_hide) + offset_hide
        # print inlayer_out
        #激活函数
        hidelayer_in = Sigmoid(inlayer_out)
        #前向过隐藏层
        hidelayer_out = np.dot(hidelayer_in, w_out) + offset_out
        #激活函数
        outlayer_in = Sigmoid(hidelayer_out)
        # 最大值索引
        index = np.argmax(outlayer_in)
        out_label.append(index)
    print (out_label)
    out_label = np.array(out_label)
    return out_label
