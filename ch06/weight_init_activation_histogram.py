"""
本节讨论权重初始值问题：
    权重初始值对于网络的拟合能力影响很大，一般遵循以下规则：
    sigmoid：Xavier 初始值 ==> 前层神经元相关  1/√n；适用于激活函数是线性函数
    ReLU：  He 初始值 ==> 前层神经元相关 √(2/n)
"""
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """
    sigmoid 函数
    :param x: 
    :return: 
    """
    return 1/(1+np.exp(-x))

def ReLU(x):
    return np.maximum(x,0)


x = np.random.randn(1000,100)
node_num = 100
hidden_layer_size = 5
activations = {}

"""
 权重初始值，标准差为1 的高斯分布
 每个网络层的输出都偏向0 or 1，会导致梯度消失，当层数增多后，进入“平原”无法梯度下降
"""
# for i in range(hidden_layer_size):
#     if i != 0:
#         x = activations[i-1]
#
#     w = np.random.randn(node_num,node_num) * 1
#     z = np.dot(x,w)
#     activations[i] = sigmoid(z)

"""
权重初始值，标准差为0.01 的高斯分布
每个网络层的输出都集中在0.5，导致每个网络输出相似没有表现力，无法提取特征
"""
# for i in range(hidden_layer_size):
#     if i != 0:
#         x = activations[i-1]
#
#     w = np.random.randn(node_num,node_num) * 0.01
#     z = np.dot(x,w)
#     activations[i] = sigmoid(z)

"""
Xavier 初始值
每个网络层的输出分布不同，表示每个网络层的表现力不同，拟合能力会更强
"""
# for i in range(hidden_layer_size):
#     if i != 0:
#         x = activations[i-1]
#
#     w = np.random.randn(node_num,node_num) * np.sqrt(1/node_num)
#     z = np.dot(x,w)
#     activations[i] = sigmoid(z)


"""
以下开始 ReLU 为激活函数
权重初始值，标准差为0.01 的高斯分布
每个网络层的输出趋向0，层数越靠后越明显
"""
# for i in range(hidden_layer_size):
#     if i != 0:
#         x = activations[i-1]
#
#     w = np.random.randn(node_num,node_num) * 0.01
#     z = np.dot(x,w)
#     activations[i] = ReLU(z)

"""
Xavier 初始值
每个网络层的输出趋向0,但相比于标准差为0.01 ，效果要好一点(为0 占比少)
"""
# for i in range(hidden_layer_size):
#     if i != 0:
#         x = activations[i-1]
#
#     w = np.random.randn(node_num,node_num) * np.sqrt(1/node_num)
#     z = np.dot(x,w)
#     activations[i] = ReLU(z)

"""
He 初始值
每层网络输出不趋向于0，
"""
for i in range(hidden_layer_size):
    if i != 0:
        x = activations[i-1]

    w = np.random.randn(node_num,node_num) * np.sqrt(2/node_num)
    z = np.dot(x,w)
    activations[i] = ReLU(z)

#  图展示
for i,value in activations.items():
    print(value.flatten())
    plt.subplot(1,len(activations),i+1)
    plt.title(str(i+1) + "layer")
    plt.hist(value.flatten(), 30, range = (0,1))

plt.show()