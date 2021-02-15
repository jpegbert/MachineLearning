import numpy as np
from scipy.spatial.distance import pdist
from sklearn.metrics import pairwise_distances
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
from math import *


"""
欧式距离python实现
"""


def oushi_distance_by_python(x, y):
    """
        使用公式计算
    """
    d1 = np.sqrt(np.sum(np.square(x - y)))
    print("d1: ", d1)


def oushi_distance_by_python1(x, y):
    """
        使用公式计算
    """
    d = sqrt(sum(pow(a - b, 2) for a, b in zip(x, y)))
    print("d: ", d)



def oushi_distance_by_numpy(x, y):
    """
    使用numpy函数计算
    """
    # 根据np.linalg.norm求解
    # np.linalg.norm是求范数函数，即先求每个元素的平方和，再开方
    d2 = np.linalg.norm(x - y)
    print("d2: ", d2)


def oushi_distance_by_scipy(x, y):
    """
    根据scipy库求解
    """
    # 将x,y两个一维数组合并成一个2D数组 ；[[x1,x2,x3...],[y1,y2,y3...]]
    X = np.vstack([x, y])
    d3 = pdist(X)
    print("d3: ", d3)
    # m * n维矩阵欧氏距离计算， 一行代表一个样本，一列代表一个特征，计算对应样本间欧氏距离
    d1 = pairwise_distances(x.reshape(-1, 10), y.reshape(-1, 10))  # 运行时间次之 占cpu多
    print("d1: ", d1)
    d2 = distance_matrix(x.reshape(-1, 10), y.reshape(-1, 10))
    print("d2: ", d2)
    d3 = cdist(x.reshape(-1, 10), y.reshape(-1, 10))  # 运行时间最短 占cpu少，建议使用
    print("d3: ", d3)


def main():
    x = np.random.random(10)
    y = np.random.random(10)
    # 使用公式计算
    oushi_distance_by_python(x, y)
    # 纯python计算
    oushi_distance_by_python1(x, y)
    # 使用numpy函数计算
    oushi_distance_by_numpy(x, y)
    # 根据scipy库求解
    oushi_distance_by_scipy(x, y)


if __name__ == '__main__':
    main()
