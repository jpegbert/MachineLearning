import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris


# 读取数据
load_iris = load_iris()
pima_df = pd.read_csv("")
print(pima_df.head())

# 读取特征、标签列，并进行中心化归一化，选取主成分个数，前2个主成分的方差和>95%
# 提取特征列
X = pima_df.iloc[:, 0:3]
# 提取标签列
y = pima_df.iloc[:, 4]
# 中心化归一化
scale = StandardScaler()
scale.fit(X)
X_scaled = scale.transform(X)
# n_components=2表示把主成分降到2维
pca = PCA(n_components=2)
x_pca = pca.fit_transform(X_scaled)
"""查看PCA的一些属性"""
# 属性可以查看降维后的每个特征向量上所带的信息量大小（可解释性方差的大小）
print(pca.explained_variance_)
# 查看降维后的每个新特征的信息量占原始数据总信息量的百分比
print(pca.explained_variance_ratio_)
# 降维后信息保留量
print(pca.explained_variance_ratio_.sum())


# 将降维后特征可视化，横纵坐标代表两个主成分，颜色代表结果标签分类，即可根据主成分进行后续分析、建模
fig = plt.figure(figsize=(6, 5))
plt.scatter(x_pca[:, 0], x_pca[: 1], c=y, s=20)
plt.xlim(-3, 3)
plt.ylim(-3, 3)
plt.xticks()
plt.yticks()
plt.show()

