import numpy as np
import scipy.stats

"""
KL散度（相对熵）python实现
"""


def kullback_leibler_divergence_by_python(x, y):
    """
    纯python实现
    """
    KL = 0.0
    px = x / np.sum(x)
    py = y / np.sum(y)
    for i in range(10):
        KL += px[i] * np.log(px[i] / py[i])
        # print(str(px[i]) + ' ' + str(py[i]) + ' ' + str(px[i] * np.log(px[i] / py[i])))
    print(KL)


def kullback__leibler_divergence_by_scipy(x, y):
    """
    利用scipy API进行计算
    """
    # scipy计算函数可以处理非归一化情况，因此这里使用
    # scipy.stats.entropy(x, y)或scipy.stats.entropy(px, py)均可
    KL = scipy.stats.entropy(x, y)
    print(KL)
    px = x / np.sum(x)
    py = y / np.sum(y)
    KL1 = scipy.stats.entropy(px, py)
    print(KL1)


if __name__ == '__main__':
    # 随机生成两个离散型分布
    x = [np.random.randint(1, 11) for i in range(10)]
    y = [np.random.randint(1, 11) for i in range(10)]
    # 纯python实现
    kullback_leibler_divergence_by_python(x, y)
    # 利用scipy API进行计算
    kullback__leibler_divergence_by_scipy(x, y)

