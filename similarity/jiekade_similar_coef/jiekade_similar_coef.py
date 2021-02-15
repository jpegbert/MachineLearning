import numpy as np
from scipy.spatial.distance import pdist


"""
杰卡德相似系数
主要用于计算符号度量或布尔值度量的样本间的相似度，等于样本集交集个数和样本集并集个数的比值。
"""


def jiekade_similar_coef_by_python(x, y):
    """
    采用纯python实现
    """
    print(x != y)
    print(np.bitwise_or(x != 0, y != 0))
    up = np.double(np.bitwise_and((x != y), np.bitwise_or(x != 0, y != 0)).sum())
    down = np.double(np.bitwise_or(x != 0, y != 0).sum())
    print("up: ", up, "down: ", down)
    d1 = (up / down)
    print("d1: ", d1)


def jiekade_similar_coef_by_python1(a, b):
    """
    采用纯python实现
    这种方式从逻辑上来看没问题，实际上与其他两种方法得到的结果不一样
    """
    unions = len(set(a).union(set(b)))
    intersections = len(set(a).intersection(set(b)))
    print("intersections: ", intersections, "unions: ", unions)
    d = float(intersections) / unions
    print("d: ", d)


def jiekade_similar_coef_by_scipy(x, y):
    """
    采用scipy库实现
    """
    X = np.vstack([x, y])
    d2 = pdist(X, 'jaccard')
    print("d2: ", d2)


def main():
    x = np.random.random(10) > 0.5
    y = np.random.random(10) > 0.5
    x = np.asarray(x, np.int32)
    y = np.asarray(y, np.int32)
    print("x: ", x)
    print("y: ", y)
    # 采用纯python实现
    jiekade_similar_coef_by_python(x, y)
    # 纯python实现
    jiekade_similar_coef_by_python1(x, y)
    # 采用scipy库实现
    jiekade_similar_coef_by_scipy(x, y)


if __name__ == '__main__':
    main()
