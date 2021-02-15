from sklearn import metrics

"""
互信息python实现
"""


def main():
    A = [1, 1, 1, 2, 3, 3]
    B = [1, 2, 3, 1, 2, 3]
    result_NMI = metrics.normalized_mutual_info_score(A, B)
    print("result_NMI:", result_NMI)


if __name__ == '__main__':
    main()
