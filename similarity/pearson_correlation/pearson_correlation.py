import numpy as np
import pandas as pd

"""
皮尔逊相关系数python实现
"""


def pearson_correlation_by_python(x, y):
    """
    纯python实现
    """
    x_ = x - np.mean(x)
    y_ = y - np.mean(y)
    d1 = np.dot(x_, y_) / (np.linalg.norm(x_) * np.linalg.norm(y_))
    print("d1: ", d1)


def pearson_correlation_by_numpy(x, y):
    """
    根据numpy库求解
    """
    X = np.vstack([x, y])
    d2 = np.corrcoef(X)[0][1]
    print("d2: ", d2)


def pearson_correlation_by_pandas(x, y):
    """
    利用pandas库求解
    """
    X1 = pd.Series(x)
    Y1 = pd.Series(y)
    d3 = X1.corr(Y1, method="pearson")
    d4 = X1.cov(Y1) / (X1.std() * Y1.std())
    print("d3: ", d3)
    print("d4: ", d4)


def main():
    x = np.random.random(10)
    y = np.random.random(10)
    # 纯python实现
    pearson_correlation_by_python(x, y)
    # 根据numpy库求解
    pearson_correlation_by_numpy(x, y)
    # 利用pandas库求解
    pearson_correlation_by_pandas(x, y)


if __name__ == '__main__':
    main()
