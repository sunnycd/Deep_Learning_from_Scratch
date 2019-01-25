import numpy as np


def softmax(a):
    """
    
    :param a: 
    :return: 
    """
    # 防止exp 指数溢出
    C = np.max(a)
    # 每类的大小
    exp_a = np.exp(a -C)
    exp_sum = np.sum(exp_a)
    return exp_a/exp_sum