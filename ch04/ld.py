"""
画出切线
"""

import numpy as np
import matplotlib.pyplot as plt

def numerical_diff(f, x):
    """
    求导函数
    :param f: 
    :param x: 
    :return: 
    """
    h = 1e-5 # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)


def function_1(x):
    """
    关于x 的曲线
    :param x: 
    :return: 
    """
    return 0.01*x**2 + 0.1*x

if __name__ == "__main__":
    """"""
    x = np.arange(0.0,20.0,0.1)
    plt.subplot(121)
    plt.plot(x,function_1(x),'--')
    ld5 = numerical_diff(function_1,5)
    # 注意这里要加上function_1(5) - 5*ld 差值，这是计算机的计算误差
    # 理论上function_1(5) = 5*ld
    plt.plot(x,x * ld5 + (function_1(5) - 5*ld5))
    plt.plot((5,5),(0,function_1(5)),'g--')
    plt.plot([0,5],[function_1(5),function_1(5)],'g--')
    plt.plot(5,function_1(5),'o')
    plt.axis([0,20,0,6])
    plt.title("5")

    plt.subplot(122)
    plt.plot(x, function_1(x), '--')
    ld10 = numerical_diff(function_1, 10)
    plt.plot(x, x * ld10 + (function_1(10) - 10 * ld10))
    plt.plot((10,10),(0,function_1(10)),'g--')
    plt.plot([0,10],[function_1(10),function_1(10)],'g--')
    plt.plot(10,function_1(10),'o')
    plt.axis([0, 20, 0, 6])
    plt.title("10")
    plt.show()

