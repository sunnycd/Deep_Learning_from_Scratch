"""
阶跃函数：
    感知机激活函数
"""

import numpy as np
import matplotlib.pyplot as plt

def step_function(x):
    # y = x > 0
    # return y.astype(np.int)
    return np.array(x>0,dtype=np.int)

if __name__ == "__main__":
    """
    
    """
    # 生成-5 -- 5 之间0.1间隔数据
    x = np.arange(-5.0,5,0.1)
    y = step_function(x)

    plt.plot(x,y)
    plt.show()

