import numpy as np
from scipy.spatial.distance import pdist

"""
布雷柯蒂斯距离python实现
"""


def bray_curtis_distance_by_python(x, y):
    """
    纯python实现
    """
    up = np.sum(np.abs(y - x))
    down = np.sum(x) + np.sum(y)
    d1 = (up / down)
    print("d1: ", d1)


def bray_curtis_distance_by_scipy(x, y):
    X = np.vstack([x, y])
    d2 = pdist(X, 'braycurtis')
    print("d2: ", d2)


def main():
    x = np.array([11, 0, 7, 8, 0])
    y = np.array([24, 37, 5, 18, 1])
    # 纯python实现
    bray_curtis_distance_by_python(x, y)
    # 采用scipy库求解
    bray_curtis_distance_by_scipy(x, y)


if __name__ == '__main__':
    main()
