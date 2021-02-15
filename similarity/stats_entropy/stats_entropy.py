import numpy as np
import scipy.stats

"""
相对熵python实现
相对熵又称KL散度（Kullback–Leibler divergence，简称KLD）
"""


def stats_entropy_by_python(p, q):
    """
    根据公式求解
    """
    kl1 = np.sum(p * np.log(p / q))
    print("kl1: ", kl1)


def stats_entropy_by_scipy(p, q):
    """
    调用scipy包求解
    """
    kl2 = scipy.stats.entropy(p, q)
    print("kl2: ", kl2)


def main():
    p = np.asarray([0.65, 0.25, 0.07, 0.03])
    q = np.array([0.6, 0.25, 0.1, 0.05])
    # 根据公式求解
    stats_entropy_by_python(p, q)
    # 调用scipy包求解
    stats_entropy_by_scipy(p, q)


if __name__ == '__main__':
    main()
