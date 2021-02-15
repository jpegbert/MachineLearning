import numpy as np
from scipy.spatial.distance import pdist


"""
闵氏距离python实现
"""


def minshi_distance_by_python(x, y):
    """
    根据公式求解, p = 2
    """
    d1 = np.sqrt(np.sum(np.square(x - y)))
    print("d1: ", d1)


def minshi_distance_by_python1(p, q, n):
    """
    纯python求解
    """
    assert len(p) == len(q)
    d = pow(sum([pow(abs(x - y), n) for x, y in zip(p, q)]),  1.0 / n)
    print("d: ", d)


def minshi_distance_by_python2(p, q, n):
    """
    纯python求解
    """
    assert len(p) == len(q)
    s = 0
    for x, y in zip(p, q):
        s += pow(abs(x - y), n)
    d = pow(s, (1.0 / n))
    print("d: ", d)


def minshi_distance_by_numpy(x, y):
    """
    采用numpy实现
    """
    d2 = np.linalg.norm(x - y, ord=2)
    print("d2: ", d2)


def minshi_distance_by_scipy(x, y):
    """
    采用scipy实现
    """
    X = np.vstack([x, y])
    d3 = pdist(X, 'minkowski', p=2)
    print("d3: ", d3)


def main():
    x = np.random.random(10)
    y = np.random.random(10)
    # 采用纯python实现
    minshi_distance_by_python(x, y)
    # 采用纯python实现，更标准
    minshi_distance_by_python1(x, y, 2)
    # 采用纯python实现，更标准
    minshi_distance_by_python2(x, y, 2)
    # 采用numpy实现
    minshi_distance_by_numpy(x, y)
    # 采用scipy实现
    minshi_distance_by_scipy(x, y)


if __name__ == '__main__':
    main()
