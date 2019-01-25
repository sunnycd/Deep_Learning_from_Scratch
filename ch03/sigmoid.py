"""
sigmoid函数：
    
"""
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

def step_function(x):
    # y = x > 0
    # return y.astype(np.int)
    return np.array(x>0,dtype=np.int)

if __name__ == "__main__":
    """
    """
    x = np.arange(-5.0,5.0,0.1)
    y = sigmoid(x)
    y1 = step_function(x)

    # 对比sigmoid 阶跃函数
    plt.plot(x,y)
    plt.plot(x,y1,'g--')
    plt.show()


