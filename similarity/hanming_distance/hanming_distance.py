import numpy as np
from scipy.spatial.distance import pdist


"""
汉明距离python实现
"""


def hanming_distance_by_python(x, y):
    """
    纯python实现
    """
    # 按照公式应该是这样的
    # d1 = np.sum(x != y)
    # 这里取了平均，也可以
    d1 = np.mean(x != y)
    print("d1: ", d1)


def hanming_distance_by_python1(x, y):
    """
    纯python实现
    """
    if len(x) != len(y):
        raise ValueError("Undefined for sequences of unequal length")
    d1 = sum(el1 != el2 for el1, el2 in zip(x, y))
    print("d1: ", d1)


def hanming_distance_by_scipy(x, y):
    """
    根据scipy库求解
    """
    X = np.vstack([x, y])
    d2 = pdist(X, 'hamming')
    print("d2: ", d2)


def main():
    x = np.random.random(10) > 0.5
    y = np.random.random(10) > 0.5
    x = np.asarray(x, np.int32)
    y = np.asarray(y, np.int32)
    # 纯python实现
    hanming_distance_by_python(x, y)
    # 纯python实现
    hanming_distance_by_python1(x, y)
    # 根据scipy库求解
    hanming_distance_by_scipy(x, y)


if __name__ == '__main__':
    main()
