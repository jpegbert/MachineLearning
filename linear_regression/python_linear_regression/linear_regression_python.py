import numpy as np
import matplotlib.pyplot as plt

"""
采用python实现线性回归的最小二乘版本
"""

points = np.genfromtxt('../data/data.csv', delimiter=',')

points[0, 0]

# 提取points中的两列数据，分别作为x，y
x = points[:, 0]
y = points[:, 1]

# 用plt画出散点图
plt.scatter(x, y)
plt.show()
