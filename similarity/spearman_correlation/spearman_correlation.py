import numpy as np
import pandas as pd

"""
斯皮尔曼相关系数python实现
"""


def spearman_correlation_by_python(x, y):
    x1 = pd.Series(x)
    y1 = pd.Series(y)
    n = x1.count()
    x1.index = np.arange(n)
    y1.index = np.arange(n)
    # 分部计算
    d = (x1.sort_values().index - y1.sort_values().index) ** 2
    dd = d.to_series().sum()
    d1 = 1 - n * dd / (n * (n ** 2 - 1))
    d2 = x1.corr(y1, method='spearman')
    print("d1: ", d1)
    print("d2: ", d2)


def main():
    x = np.random.random(10)
    y = np.random.random(10)
    spearman_correlation_by_python(x, y)


if __name__ == '__main__':
    main()
