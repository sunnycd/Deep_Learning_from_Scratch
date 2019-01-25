"""
感知机：
    
"""

import numpy as np

def OR(x1,x2):
    """
    不同的权重值，就可以生成不同的模型，或、与、与非
    :param x1: 
    :param x2: 
    :return: 
    """
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def NAND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([-0.5,-0.5])
    b = 0.7
    # 对比AND，NAND就是权重符号取反
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def AND(x1,x2):
    """ x1 and x2 """
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    # b 为偏置
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


def XOR(x1,x2):
    """
    或、与非、与 实现是线性可分只需单层感知机
    异或是非线性，多要多层感知机
    :param x1: 
    :param x2: 
    :return: 
    """
    # 下面2个感知机同层
    s1 = NAND(x1,x2)
    s2 = OR(x1,x2)
    # 这是第二层感知机
    y = AND(s1,s2)

    return y

if __name__ == "__main__":
    """
    
    """

