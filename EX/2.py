import numpy as np
import random

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

def hill_climbing(max_iter=1000):
    # 初始路徑
    current_path = list(range(len(citys)))  
    current_length = path_length(current_path)

    for _ in range(max_iter):
        # 隨機交換兩個城市位置，形成鄰近解
        i, j = random.sample(range(len(citys)), 2)
        current_path[i], current_path[j] = current_path[j], current_path[i]
        new_length = path_length(current_path)

        # 如果新解更優，則接受新解
        if new_length < current_length:
            current_length = new_length
        else:
            # 恢復原來的路徑
            current_path[i], current_path[j] = current_path[j], current_path[i]

    return current_path, current_length

best_path, best_length = hill_climbing()
print('Best Path:', best_path)
print('Best Length:', best_length)
