"""
本节利用 mnist 数据集验证上节介绍的权重初始化
    高斯分布方差为 0.01，在 2000 次后 loss 可以说没有下降，无法优化
    He 比 Xavier 下降速度更快，且优化更好(激活函数是 ReLU)
"""
import numpy as np
from dataset.mnist import load_mnist
from layer import optimizer
from layer import multi_layer_net
import common
import matplotlib.pyplot as plt

optimizers = optimizer.SGD(0.01)
weight_init_types = {'std=0.01': 0.01, 'Xavier': 'sigmoid', 'He': 'relu'}

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)


networks = {}
train_loss = {}
# 初始化网络
for key in weight_init_types.keys():
    # 5 层网络
    networks[key] = multi_layer_net.MultiLayerNet(input_size=784,
                                                  hidden_size_list=[100,100,100,100],
                                                  output_size=10,weight_init_std = weight_init_types[key])
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
    for key in weight_init_types.keys():
        grads = networks[key].gradient(train_batch,label_batch)
        optimizers.optimizer(networks[key].params,grads)

        loss = networks[key].loss(train_batch,label_batch)
        train_loss[key].append(loss)

        if i%100 == 0:
            print(key +" loss : "+str(loss))

# 画图
markers = {"std=0.01": "o-", "Xavier": "x-", "He": "s-"}
x = np.arange(max_iterations)

for key in weight_init_types.keys():
    plt.plot(x,train_loss[key],markers[key],label = key)

plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 3)
plt.legend()
plt.show()