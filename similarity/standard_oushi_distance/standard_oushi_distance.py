import numpy as np
from scipy.spatial.distance import pdist


"""
标准化欧式距离实现
"""


def standard_oushi_distance_by_python(x, y):
    """
    采用纯python实现
    """
    X = np.vstack([x, y])
    sk = np.var(X, axis=0, ddof=1)
    d1 = np.sqrt(((x - y) ** 2 / sk).sum())
    print("d1: ", d1)


def standard_oushi_distance_by_python1(a, b):
    sumnum = 0
    for i in range(len(a)):
        avg = (a[i] + b[i]) / 2
        si = ((a[i] - avg) ** 2 + (b[i] - avg) ** 2) ** 0.5
        sumnum += ((a[i] - b[i]) / si) ** 2
    d = sumnum ** 0.5
    print("d: ", d)


def standard_oushi_distance_by_numpy(x, y):
    """
    采用numpy实现
    """
    X = np.vstack([x, y])
    sk = np.var(X, axis=0, ddof=1)
    d2 = np.linalg.norm((x - y) / np.sqrt(sk), ord=2)
    print("d2: ", d2)


def standard_oushi_distance_by_scipy(x, y):
    """
    采用scipy实现
    """
    X = np.vstack([x, y])
    d3 = pdist(X, 'seuclidean', [0.5, 1])
    print("d3: ", d3)


def main():
    x = np.random.random(10)
    y = np.random.random(10)
    # 采用纯python实现
    standard_oushi_distance_by_python(x, y)
    # 纯python实现
    standard_oushi_distance_by_python1(x, y)
    # 采用numpy实现
    standard_oushi_distance_by_numpy(x, y)
    # 采用scipy实现
    standard_oushi_distance_by_scipy(x, y)


if __name__ == '__main__':
    main()
