import numpy as np
from scipy.spatial.distance import pdist

"""
马氏距离python实现
"""


def mashi_distance_by_python(x, y):
    """
    纯python实现
    """
    # 马氏距离要求样本数要大于维数，否则无法求协方差矩阵
    # 此处进行转置，表示10个样本，每个样本2维
    X = np.vstack([x, y])
    XT = X.T
    S = np.cov(X)  # 两个维度之间协方差矩阵
    SI = np.linalg.inv(S)  # 协方差矩阵的逆矩阵
    # 马氏距离计算两个样本之间的距离，此处共有10个样本，两两组合，共有45个距离。
    n = XT.shape[0]
    d1 = []
    for i in range(0, n):
        for j in range(i + 1, n):
            delta = XT[i] - XT[j]
            d = np.sqrt(np.dot(np.dot(delta, SI), delta.T))
            d1.append(d)
    print("d1: ", d1)


def mashi_distance_by_scipy(x, y):
    """
    采用scipy实现
    """
    # 马氏距离要求样本数要大于维数，否则无法求协方差矩阵
    # 此处进行转置，表示10个样本，每个样本2维
    X = np.vstack([x, y])
    XT = X.T
    d2 = pdist(XT, 'mahalanobis')
    print("d2: ", d2)


def main():
    x = np.random.random(10)
    y = np.random.random(10)
    # 纯python实现
    mashi_distance_by_python(x, y)
    # 采用scipy库实现
    mashi_distance_by_scipy(x, y)


if __name__ == '__main__':
    main()
