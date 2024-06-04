# micrograd/engine.py
import numpy as np
from numpy.linalg import norm
from micrograd.engine import Value

def gradientDescendent(f, p0, h=0.01, max_loops=100000, dump_period=1000):
    p = [Value(x) for x in p0]
    for i in range(max_loops):
        for v in p:
            v.grad = 0.0
        fp = f(p)
        fp.backward()
        gp = np.array([v.grad for v in p])
        glen = norm(gp)
        if i % dump_period == 0:
            print('{:05d}:f(p)={:.3f} p={:s} gp={:s} glen={:.5f}'.format(i, fp.data, str([v.data for v in p]), str(gp), glen))
        if glen < 0.00001:
            break
        gh = np.multiply(gp, -1 * h)
        for j in range(len(p)):
            p[j].data += gh[j]
    print('{:05d}:f(p)={:.3f} p={:s} gp={:s} glen={:.5f}'.format(i, fp.data, str([v.data for v in p]), str(gp), glen))
    return [v.data for v in p]

# 測試案例
import matplotlib.pyplot as plt

# 數據集
x = np.array([0, 1, 2, 3, 4], dtype=np.float32)
y = np.array([1.9, 3.1, 3.9, 5.0, 6.2], dtype=np.float32)

# 預測函數
def predict(a, xt):
    return a[0] + a[1] * xt

# 均方誤差函數
def MSE(a, x, y):
    total = Value(0.0)
    for i in range(len(x)):
        total += (y[i] - predict(a, x[i])) ** 2
    return total

# 損失函數
def loss(p):
    return MSE(p, x, y)

# 初始參數
p = [0.0, 0.0]

# 梯度下降
plearn = gradientDescendent(loss, p, max_loops=3000, dump_period=1000)

# 繪製
y_predicted = [plearn[0] + plearn[1] * t for t in x]
print('y_predicted=', y_predicted)
plt.plot(x, y, 'ro', label='Original data')
plt.plot(x, y_predicted, label='Fitted line')
plt.legend()
plt.show()
