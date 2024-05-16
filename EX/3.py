from scipy.optimize import linprog

# 定義目標函數係數
c = [-3, -2, -5]  # 最大化，所以係數加負號

# 定義限制條件的係數矩陣和右側常數
A = [[1, 1, 0], [2, 0, 1], [0, 1, 2]]
b = [10, 9, 11]

# 定義變數的範圍
x_bounds = (0, None)
y_bounds = (0, None)
z_bounds = (0, None)

# 使用linprog函數，不指定求解器，讓函數自動選擇
res = linprog(c, A_ub=A, b_ub=b, bounds=[x_bounds, y_bounds, z_bounds])

# 輸出結果
print('最大值:', -res.fun)  # 直接使用最佳值的負值來得到最大值
print('最佳解:', res.x)  # 最佳解的x、y、z值
