from sklearn.datasets import fetch_lfw_people
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


# 导入数据，并且探索一下子
faces = fetch_lfw_people(min_faces_per_person=60)
print(faces.images.shape)   # (1348, 64, 47)  1348张图片，每张64*47
print(faces.data.shape)    # (1348, 2914)  这是把上面的后两维进行了合并，共2914个特征（像素点）
# 下面我们先可视化一下子这些图片，看看长什么样
fig, axes = plt.subplots(3, 8, figsize=(8, 4), subplot_kw={"xticks": [], "yticks": []})

for i, ax in enumerate(axes.flat):
    ax.imshow(faces.images[i, :, :], cmap='gray')

pca = PCA(150).fit(faces.data)  # 降到150维
V = pca.components_   # 这就是那组基
print(V.shape) #（150,2914） 每一行是一个基，用这个乘上我们样本X，就会得到降维后的结果矩阵

# 下面可视化一下V
fig, axes = plt.subplots(3, 8, figsize=(8, 4), subplot_kw={"xticks": [], "yticks": []})
for i, ax in enumerate(axes.flat):
    ax.imshow(V[i, :].reshape(62, 47), cmap='gray')

# 我们先得到降维后的数据
X_dr = pca.transform(faces.data)    # 这个是1358,150的数据

# 然后我们调用接口逆转
X_inverse = pca.inverse_transform(X_dr)
print(X_inverse.shape)    # （1348， 2914） 看这个形状还真回去了啊

# 下面对比一下pca的逆转和原来图片的区别
fig, ax = plt.subplots(2, 10, figsize=(10, 2.5), subplot_kw={"xticks": [], "yticks": []})
for i in range(10):
    ax[0, i].imshow(faces.images[i, :, :], cmap='binary_r')
    ax[1, i].imshow(X_inverse[i].reshape(62, 47), cmap="binary_r")   # 降维不是完全可逆的



