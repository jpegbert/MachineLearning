import numpy as np
from scipy.spatial.distance import pdist

"""
切比雪夫距离python实现
"""


def qiebixuefu_distance_by_python(x, y):
    """
    纯python实现
    """
    d1 = np.max(np.abs(x - y))
    print("d1: ", d1)


def qiebixuefu_distance_by_python1(p, q):
    """
    纯python实现
    """
    assert len(p) == len(q)
    d = max([abs(x - y) for x, y in zip(p, q)])
    print("d: ", d)


def qiebixuefu_distance_by_python2(p, q):
    assert len(p) == len(q)
    d = 0
    for x, y in zip(p, q):
        d = max(d, abs(x - y))
    print("d: ", d)


def qiebixuefu_distance_by_numpy(x, y):
    """
    采用numpy库实现
    """
    d2 = np.linalg.norm(x - y, ord=np.inf)
    print("d2: ", d2)


def qiebixuefu_distance_by_scipy(x, y):
    """
    采用scipy库实现
    """
    X = np.vstack([x, y])
    d3 = pdist(X, 'chebyshev')
    print("d3: ", d3)


def main():
    x = np.random.random(10)
    y = np.random.random(10)
    # 纯python实现
    qiebixuefu_distance_by_python(x, y)
    # 纯python实现
    qiebixuefu_distance_by_python1(x, y)
    # 纯python实现
    qiebixuefu_distance_by_python2(x, y)
    # 采用numpy库实现
    qiebixuefu_distance_by_numpy(x, y)
    # 采用scipy库实现
    qiebixuefu_distance_by_scipy(x, y)


if __name__ == '__main__':
    main()
