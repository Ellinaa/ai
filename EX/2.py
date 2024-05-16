import numpy as np

citys = np.array([
    (0,3),(0,0),
    (0,2),(0,1),
    (1,0),(1,3),
    (2,0),(2,3),
    (3,0),(3,3),
    (3,1),(3,2)
])

# 計算城市之間的距離矩陣
distances = np.linalg.norm(citys[:, np.newaxis, :] - citys[np.newaxis, :, :], axis=-1)

def path_length(p):
    return sum(distances[p[i], p[(i+1) % len(p)]] for i in range(len(p)))

# 初始路徑
path = list(range(len(citys)))  

print('pathLength =', path_length(path))
