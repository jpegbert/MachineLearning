import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


iris = load_iris()
X = iris.data   # Xshape(150, 4)

# X的归一化
X_norm = StandardScaler().fit_transform(X)
X_norm.mean(axis=0)      # 这样每一维均值为0了

# 求特征值和特征向量
# np.cov直接求协方差矩阵，每一行代表一个特征，每一列代表样本
ew, ev = np.linalg.eig(np.cov(X_norm.T))
print(ew)
print(ev)

# 特征向量特征值的排序
ew_oreder = np.argsort(ew)[::-1]
print("ew_oreder", ew_oreder)
ew_sort = ew[ew_oreder]
print("ew_sort", ew_sort)
ev_sort = ev[:, ew_oreder]  # ev的每一列代表一个特征向量
print("ev_sort", ev_sort)
print(ev_sort.shape) # (4,4)

# 我们指定降成2维， 然后取出排序后的特征向量的前两列就是基
K = 2
V = ev_sort[:, :2]  # 4*2

# 最后，我们得到降维后的数据
X_new = X_norm.dot(V)    # shape (150,2)

colors = ['red', 'black', 'orange']
plt.figure()
for i in [0, 1, 2]:
    plt.scatter(X_new[iris.target==i, 0],
                X_new[iris.target==i, 1],
                alpha=.7,
                c=colors[i],
                label=iris.target_names[i]
               )

plt.legend()
plt.title('PCa of IRIS dataset')
plt.xlabel('PC_0')
plt.ylabel('PC_1')
plt.show()



