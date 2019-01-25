"""
三层神经网络
"""
import numpy as np

def sigmoid(x):
    return 1/(1+np.exp(-x))

def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network

def forward(network, x):
    """
    前向处理
    :param network: 
    :param x: 
    :return: 
    """
    y1 = np.dot(x , network["W1"])  + network['b1']
    y2 =  np.dot(sigmoid(y1) , network["W2"]) + network['b2']
    y3 =  np.dot(sigmoid(y2) , network["W3"]) + network['b3']
    return y3

if __name__ == "__main__":
    """
    """
    network = init_network()
    x = np.array([1.0,0.5])
    y = forward(network,x)
    print(y)

