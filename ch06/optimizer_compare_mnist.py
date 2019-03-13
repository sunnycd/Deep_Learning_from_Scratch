"""
本节讨论，loss 优化过程中的梯度下降算法
    很明显，Adam 下降最快，SGD 在2000 后还没接近最优解，
    是不是往后训练使用 Adam 就可以了。不尽然，每个 optimizer 都有自己的特点，
    要根据数据来选择合适的optimizer
    Adam那么棒，为什么还对SGD念念不忘 (2)—— Adam的两宗罪(https://zhuanlan.zhihu.com/p/32338983)
"""
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
from layer import optimizer
from layer import multi_layer_net
import common
import matplotlib.pyplot as plt


optimizers = {}
optimizers['SGD'] = optimizer.SGD()
optimizers['Momentum'] = optimizer.Momentum()
optimizers['AdaGrad'] = optimizer.AdaGrad()
optimizers['Adam'] = optimizer.Adam()

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)


networks = {}
train_loss = {}
# 初始化网络
for key in optimizers.keys():
    # 5 层网络
    networks[key] = multi_layer_net.MultiLayerNet(input_size=784,
                                                  hidden_size_list=[100,100,100,100],
                                                  output_size=10)
    train_loss[key] = []

train_size = x_train.shape[0]
batch_size = 128
max_iterations = 2000
# 训练
for i in range(max_iterations):
    batch_mask = np.random.choice(train_size,batch_size)
    train_batch = x_train[batch_mask]
    label_batch = t_train[batch_mask]

    if i%100 == 0:
        print("*********** iteration: "+str(i)+" **********************")
    for key in optimizers.keys():
        grads = networks[key].gradient(train_batch,label_batch)
        optimizers[key].optimizer(networks[key].params,grads)

        loss = networks[key].loss(train_batch,label_batch)
        train_loss[key].append(loss)

        if i%100 == 0:
            print(key +" loss : "+str(loss))

# 画图
markers = {"SGD": "o-", "Momentum": "x-", "AdaGrad": "s-", "Adam": "D-"}
x = np.arange(max_iterations)

for key in optimizers.keys():
    plt.plot(x,train_loss[key],markers[key],label = key)

plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 3)
plt.legend()
plt.show()

