import numpy as np
from mnistsdataset.mnist import load_mnist
import matplotlib
import matplotlib.pyplot as plt
import pickle
from sklearn import metrics

def sigmoid(x):
    return 1/(1+np.exp(-x))

def loadimg(x):
    # cmap 颜色种类，binary 黑白
    plt.imshow(x,cmap = matplotlib.cm.binary,interpolation="nearest")

def softmax(a):
    """

    :param a: 
    :return: 
    """
    # 防止exp 指数溢出
    C = np.max(a)
    # 每类的大小
    exp_a = np.exp(a - C)
    exp_sum = np.sum(exp_a)
    return exp_a / exp_sum

def get_data():
    """
    
    :return: 
    """
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    """
    
    :return: 
    """
    with open("sample_weight.pkl","rb") as f:
        network = pickle.load(f)
    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    return y

if __name__ =="__main__":
    """"""
    # (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True,normalize=True)
    x,y = get_data()
    # 事先训练好的神经网络，第一层50个神经元，第二层100个神经元
    network = init_network()

    # 每批训练数据100个
    batch_size = 100
    accuracy_cnt = 0
    # for i in range(len(x)):
    #     y_pred = predict(network, x[i])
    #     p = np.argmax(y_pred)  # 获取概率最高的元素的索引
    #     if p == y[i]:
    #         accuracy_cnt += 1

    for i in range(0,len(x),batch_size):
        y_pred_batch = predict(network,x[i:i+batch_size])
        # 解释下为什么axis=1，求每行的最大值，就是比较同一行的每一列，所以axis=1 表示列
        p_batch = np.argmax(y_pred_batch,axis=1)
        accuracy_cnt += np.sum(p_batch == y[i:i+batch_size])

    print("Accuracy:" + str(float(accuracy_cnt) / len(x)))