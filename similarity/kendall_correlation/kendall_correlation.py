import pandas as pd


"""
肯德尔相关性系数python实现
"""


def main():
    # 原始数据
    x = pd.Series([3, 1, 4, 2, 5, 3])
    y = pd.Series([1, 2, 3, 2, 1, 1])
    d = x.corr(y, method="kendall")
    print("d: ", d)


if __name__ == '__main__':
    main()
