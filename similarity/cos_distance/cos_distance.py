import numpy as np
from scipy.spatial.distance import pdist
from math import *


"""
余弦相似度python实现
"""


def cos_distance_by_python(x, y):
    """
    纯python实现
    """
    d1 = np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))
    print("d1: ", d1)


def square_rooted(x):
    return round(sqrt(sum([a * a for a in x])), 6)


def cos_distance_by_python1(x, y):
    """
    纯python实现
    """
    numerator = sum(a * b for a, b in zip(x, y))
    denominator = square_rooted(x) * square_rooted(y)
    d = round(numerator / float(denominator), 6)
    print("d: ", d)


def cos_distance_by_scipy(x, y):
    """
    根据scipy库求解
    """
    X = np.vstack([x, y])
    d2 = 1 - pdist(X, 'cosine')
    print("d2: ", d2)


def main():
    x = np.random.random(10)
    y = np.random.random(10)
    # 纯python实现
    cos_distance_by_python(x, y)
    # 纯python实现
    cos_distance_by_python1(x, y)
    # 根据scipy库求解
    cos_distance_by_scipy(x, y)


if __name__ == '__main__':
    main()

