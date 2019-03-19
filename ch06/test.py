import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.util import im2col

# x1 = np.random.rand(1, 3, 7, 7)
# print(x1)
# col1 = im2col(x1, 5, 5, stride=1, pad=0)
#
# print(col1.shape) # (9, 75)
#
# x2 = np.random.rand(10, 3, 7, 7) # 10个数据
# col2 = im2col(x2, 5, 5, stride=1, pad=0)
# print(col2.shape) # (90, 75)

x1 = np.arange(1,19).reshape(1,2,3,3)
print(x1)
col1 = im2col(x1,3,3,stride=1,pad=1)
print(col1)

a = np.arange(6).reshape(2,3)
print(np.max(a[1]))
print(np.argmax(a))
print(np.argmax(a,axis=0))
print(np.argmax(a,axis=1))