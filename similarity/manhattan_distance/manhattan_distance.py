import numpy as np
from scipy.spatial.distance import pdist


"""
曼哈顿距离python实现
"""


def manhattan_distance_by_python(x, y):
    """
    纯python实现曼哈顿距离
    """
    d1 = np.sum(np.abs(x - y))
    print("d1: ", d1)


def manhattan_distance_by_python1(x, y):
    """
    纯python实现
    """
    d1 = sum(abs(a-b) for a, b in zip(x, y))
    print("d1: ", d1)


def manhattan_distance_by_numpy(x, y):
    """
    采用numpy实现曼哈顿距离
    """
    d2 = np.linalg.norm(x - y, ord=1)
    print("d2: ", d2)


def manhattan_distance_by_scipy(x, y):
    """
    采用scipy库实现曼哈顿距离
    """
    X = np.vstack([x, y])
    d3 = pdist(X, 'cityblock')
    print("d3: ", d3)


def main():
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6])
    # 纯python实现曼哈顿距离
    manhattan_distance_by_python(x, y)
    # 纯python实现
    manhattan_distance_by_python1(x, y)
    # 采用numpy实现曼哈顿距离
    manhattan_distance_by_numpy(x, y)
    # 采用scipy库实现曼哈顿距离
    manhattan_distance_by_scipy(x, y)


if __name__ == '__main__':
    main()
