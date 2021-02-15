import numpy as np
from scipy.stats import chisquare


"""
卡方检验python实现

卡方公式(o-e)^2 / e
期望值和收集到数据不能低于5，o(observed)观察到的数据，e（expected）表示期望的数据
(o-e)平方，最后除以期望的数据e
"""


def chisquare_check_by_python(list_observe, list_expect):
    """
    根据公式求解（最后根据c1的值去查表判断）
    """
    print("--", np.square(list_observe - list_expect))
    c1 = np.sum(np.square(list_observe - list_expect) / list_expect)
    print("c1: ", c1)


def chisquare_check_by_scipy(list_observe, list_expect):
    """
    使用scipy库来求解
    """
    c2, p = chisquare(f_obs=list_observe, f_exp=list_expect)
    print("c2: ", c2)
    if p > 0.05 or p == "nan": # 返回NAN，无穷小
        print("H0 win,there is no difference")
    else:
        print("H1 win,there is difference")


def main():
    list_observe = np.array([30, 14, 34, 45, 57, 20])
    list_expect = np.array([20, 20, 30, 40, 60, 30])
    # 根据公式求解
    chisquare_check_by_python(list_observe, list_expect)
    # 使用scipy库来求解
    chisquare_check_by_scipy(list_observe, list_expect)


if __name__ == '__main__':
    main()
