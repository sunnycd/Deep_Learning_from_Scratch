""""
显示  x1**2 + x0**2 三维图
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def function2(X,y):
    """
    X**2 + y**2
    :param X: 
    :param y: 
    :return: 
    """
    return X**2 + y**2

if __name__ == "__main__":
    """
    """
    x0 = np.arange(-2,2.5,0.25)
    x1 = np.arange(-2,2.5,0.25)
    X,y = np.meshgrid(x0,x1)

    ax = plt.axes(projection='3d')
    # 3d
    ax.plot_wireframe(X,y,function2(X,y))
    plt.xlabel("X")
    plt.ylabel("y")
    plt.show()



