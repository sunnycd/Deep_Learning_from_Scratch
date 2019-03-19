"""
优化相关类：SGD、Momentum、AdaGrad、RMSpop、Adam(AdaGrad + Mometum)
"""


import numpy as np

class SGD:
    """
    随机梯度下降，特征要做归一化，不然优化过程走"之"，耗费时间
    """
    def __init__(self,lr = 0.001):
        """
        
        :param lr: 学习率
        """
        self.lr = lr

    def optimizer(self,params,grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]


class Momentum:
    """
    在 SGD 基础上增加 Momentum，加快下降速度
    当前下降方向与上次下降一致，比方向不一致的速度要快
    self.momentum * self.v[key] - self.lr * grads[key]
    momentum 动量，在当前力与运动方向相同时，速度加快；当前力与运动方向相反时，速度减少
    使优化目标性更强，避免 SGD  的“之” 下降，从而减少整体优化时间
    """
    def __init__(self,lr = 0.001,momentum = 0.9):
        """
        
        :param lr: 
        :param mometum: 
        """
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def optimizer(self,params,grads):
        """
        
        :param params: 
        :param grads: 
        :return: 
        """
        if self.v == None:
            self.v = {}
            for key,val in params.items():
                # 初始化全为0
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]


class Nesterov:
    """Nesterov's Accelerated Gradient (http://arxiv.org/abs/1212.0901)"""

    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)

        for key in params.keys():
            self.v[key] *= self.momentum
            self.v[key] -= self.lr * grads[key]
            params[key] += self.momentum * self.momentum * self.v[key]
            params[key] -= (1 + self.momentum) * self.lr * grads[key]

class AdaGrad:
    """
    梯度下降过程中，刚开始下降速度快，接近最优点时应该降低速度
    AdaGrad 就是深入了以上想法，计算每个权重以往斜率之后来得到当前权重斜率系数
    显然，训练越久，斜率系数越小，下降越慢
    AdaGrad 很大程度解决下降过程在两边相互跳跃的情形
    但当 optimizer 次数越来越多时，斜率系数会趋向于0 导致无法更新权重，
    为了避免这种情况，RMSprop 出现了，它不会累加过往所有而是慢慢抛弃之前着重最新
    """
    def __init__(self,lr= 0.001,):
        self.lr = lr
        self.h = None

    def optimizer(self,params,grads):
        """
        
        :param params: 
        :param grads: 
        :return: 
        """
        if self.h == None:
            self.h = {}
            for key ,val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr*grads[key] / np.sqrt(self.h[key] + 1e-7)


class RMSprop:
    """RMSprop"""

    def __init__(self, lr=0.01, decay_rate=0.99):
        self.lr = lr
        self.decay_rate = decay_rate
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)

        for key in params.keys():
            # 历史占 90%，当前占10%
            self.h[key] *= self.decay_rate
            self.h[key] += (1 - self.decay_rate) * grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)

class Adam:
    """
    Adam (http://arxiv.org/abs/1412.6980v8)
    Adam 结合 AdaGrad 和 Mometum 优点
    https://zhuanlan.zhihu.com/p/32338983 阐述Adam 缺点
    总结来说：Adam 设计想法很智能，但深度学习网络太复杂，Adam 往往无法适应(理想很丰满，现实很骨感)
    但在稀疏网络下，Aadm 确实能加快寻找最优解途径，但往往得不到最优解
    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        """
        三个超参
        :param lr: 
        :param beta1: 
        :param beta2: 
        """
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None

    def optimizer(self, params, grads):
        if self.m is None:
            self.m, self.v = {}, {}
            for key, val in params.items():
                self.m[key] = np.zeros_like(val)
                self.v[key] = np.zeros_like(val)

        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2 ** self.iter) / (1.0 - self.beta1 ** self.iter)

        for key in params.keys():
            # self.m[key] = self.beta1*self.m[key] + (1-self.beta1)*grads[key]
            # self.v[key] = self.beta2*self.v[key] + (1-self.beta2)*(grads[key]**2)
            self.m[key] += (1 - self.beta1) * (grads[key] - self.m[key])
            self.v[key] += (1 - self.beta2) * (grads[key] ** 2 - self.v[key])

            params[key] -= lr_t * self.m[key] / (np.sqrt(self.v[key]) + 1e-7)

            # unbias_m += (1 - self.beta1) * (grads[key] - self.m[key]) # correct bias
            # unbisa_b += (1 - self.beta2) * (grads[key]*grads[key] - self.v[key]) # correct bias
            # params[key] += self.lr * unbias_m / (np.sqrt(unbisa_b) + 1e-7)