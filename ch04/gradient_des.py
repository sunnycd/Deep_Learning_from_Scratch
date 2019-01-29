"""
梯度下降示意图：
    函数f(x)=x0**2 + x1**2，梯度下降图，初始值为(3,4)，理想是找到最优值(0，0)
"""
import numpy as np
import matplotlib.pyplot as plt


def function(x):
    """"
    x 平方和
    """
    if x.ndim == 1:
       return np.sum(x**2)
    else:
        return np.sum(x**2,axis=1)

def numerical_gradient(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x) # 生成和x形状相同的数组

    for idx in range(x.size):
        tmp_val = x[idx]
        # f(x+h)的计算
        x[idx] = tmp_val + h
        fxh1 = f(x)

        # f(x-h)的计算
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val # 还原值

    return grad

def gradient_descent(f,init,lr=0.1,step_num=100):
    """
    
    :param f: 
    :param init: 
    :param lr: 
    :param step_num: 
    :return: 
    """
    x = init
    gradients = np.zeros((step_num+1,2))
    gradients[0] = x
    for i in range(step_num):
        gradient = numerical_gradient(f,x)
        x -= lr*gradient
        gradients[i+1] = x
    return gradients

def plt_circle(x,r):
    """
    
    :param x: x 取值范围
    :param r: 半径
    :return: y 取值范围
    """
    return  np.sqrt(r**2 - x**2)

def circle(x,y,r,color='k',count=1000):
    """
    画圆
    :param x: 圆心
    :param y: 圆心
    :param r: 半径
    :param color: 
    :param count: 
    :return: 
    """
    xarr=[]
    yarr=[]
    for i in range(count):
        j = float(i)/count * 2 * np.pi
        xarr.append(x+r*np.cos(j))
        yarr.append(y+r*np.sin(j))

    plt.plot(xarr, yarr, '--',c=color)

if __name__ == "__main__":
    """
    
    """
    init = np.array([-3.0,4.0])
    grad_point = gradient_descent(function,init)
    X = np.arange(-2,2,0.1)
    y = np.arange(-2,2,0.2)

    plt.axis([-4.5,4.5,-4.5,4.5])
    plt.plot(grad_point[:,0],grad_point[:,1],'o')
    circle(0, 0, 0.5)
    circle(0, 0, 1)
    circle(0,0,2)
    circle(0, 0, 3)
    circle(0,0,4)
    plt.show()
