"""
接下来两节讲解过拟合的处理方法：
    在机器学习中，模型过拟合原因如下：
        模型太复杂：简化模型，弱化拟合能力；重新特征工程，选择合适特征；正则化惩罚模型
        数据量少：增大训练数据
    在深度学习中也是类似，本节是选用 L2 正则，下节 Dropout 是类似机器学习中的集成方法，典型代表随机森林。
    加 L2 正则和没加的模型都输出，你可以明显看出没加 L2 的模型，
在训练一段时间后，train 准确率提高的同时 test 准确率降低了，说明过拟合
"""
import numpy as np
from dataset.mnist import load_mnist
from layer import optimizer
from layer import multi_layer_net
import common
import matplotlib.pyplot as plt

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)
# 为了再现过拟合，减少学习数据
x_train = x_train[:300]
t_train = t_train[:300]

# weight decay（权值衰减）的设定
weight_decays = {}
weight_decays["weight_decay_lambda"] = 0.1
weight_decays["weight_decay_no"] = 0

networks = {}
train_loss_lists = {}
train_acc_lists = {}
test_acc_lists = {}

optimizer = optimizer.SGD(lr=0.01)
for key,val in weight_decays.items():
    networks[key] = multi_layer_net.MultiLayerNet(input_size=784, hidden_size_list=[100, 100, 100, 100, 100, 100], output_size=10,
                            weight_decay_lambda=val)
    train_loss_lists[key] = []
    train_acc_lists[key] = []
    test_acc_lists[key] = []

max_epochs = 100
train_size = x_train.shape[0]
batch_size = 100



iter_per_epoch = max(train_size / batch_size, 1)
epoch_cnt = 0

for i in range(10000):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    for key in weight_decays.keys():

        grads = networks[key].gradient(x_batch, t_batch)
        optimizer.optimizer(networks[key].params, grads)

    if i % iter_per_epoch == 0:
        for key in weight_decays.keys():
            train_acc = networks[key].accuracy(x_train, t_train)
            test_acc = networks[key].accuracy(x_test, t_test)
            train_acc_lists[key].append(train_acc)
            test_acc_lists[key].append(test_acc)

            print(str(key) + " epoch:" + str(epoch_cnt) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc))

        epoch_cnt += 1
        if epoch_cnt >= max_epochs:
            break


# 3.绘制图形==========
markers = {'train': 'o', 'test': 's'}
x = np.arange(max_epochs)
for key in weight_decays.keys():
    if(key == "weight_decay_no"):
        plt.subplot(2, 1, 1)
    else:
        plt.subplot(2, 1,2)
    plt.title(key)
    plt.plot(x, train_acc_lists[key], marker='o', label='train', markevery=10)
    plt.plot(x, test_acc_lists[key], marker='s', label='test', markevery=10)
    plt.xlabel("epochs")
    plt.ylabel("accuracy")
    plt.ylim(0, 1.0)
    plt.legend(loc='lower right')

plt.show()