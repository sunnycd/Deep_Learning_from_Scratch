"""
Rectified Linear Unit:
    ReLU 函数在输入大于 0 时，直接输出该值；在输入小于等于 0 时，输出 0
"""

import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(x,0)

if __name__ == "__main__":
    """
    """
    x = np.arange(-2.0,5.0,0.1)
    y = relu(x)

    plt.plot(x,y)
    plt.show()