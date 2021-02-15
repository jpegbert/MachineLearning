import numpy as np


"""
DTW距离实现
DTW（Dynamic Time Warping，动态时间归整）是一种衡量两个长度不等的时间序列间相似度的方法，
主要应用在语音识别领域，识别两段语音是否表示同一个单词。
"""


def cal_dtw_distance(X, Y):  # dtw距离计算
    sign_len_N, num_features = X.shape               # 获取T的行数features，和列数N
    sign_len_M, num_features = Y.shape[1]           # 获取R的列数
    eudist_matrix = np.zeros((sign_len_N, sign_len_M))
    for i in range(num_features):                              # 生成原始距离矩阵
        eudist_matrix += pow(np.transpose([X[i, :]]) - Y[i, :], 2)
    eudist_matrix = np.sqrt(eudist_matrix)
    # 动态规划
    dtw_distance_matrix = np.zeros(np.shape(eudist_matrix))
    dtw_distance_matrix[0, 0] = eudist_matrix[0, 0]
    for n in range(1, sign_len_N):
        dtw_distance_matrix[n, 0] = eudist_matrix[n, 0] + dtw_distance_matrix[n-1, 0]
    for m in range(1, sign_len_M):
        dtw_distance_matrix[0, m] = eudist_matrix[0, m] + dtw_distance_matrix[0, m-1]
    # 三个方向最小
    for n in range(1, sign_len_N):
        for m in range(1, sign_len_M):
            dtw_distance_matrix[n, m] = eudist_matrix[n, m] + \
                min([dtw_distance_matrix[n-1, m], dtw_distance_matrix[n-1, m-1], dtw_distance_matrix[n, m-1]])  # 动态计算最短距离
    n = sign_len_N-1
    m = sign_len_M-1
    k = 1
    warping_path = [[sign_len_N-1, sign_len_M-1]]
    while n+m != 0:  # 匹配路径过程
        if n == 0:
            m = m-1
        elif m == 0:
            n = n-1
        else:
            number = np.argmin([dtw_distance_matrix[n-1, m], dtw_distance_matrix[n-1, m-1], dtw_distance_matrix[n, m-1]])
            if number == 0:
                n = n-1
            elif number == 1:
                n = n-1
                m = m-1
            elif number == 2:
                m = m-1
        k = k+1
        warping_path.append([n, m])
    warping_path = np.array(warping_path)
    dtw_distance = dtw_distance_matrix[-1, -1]  # 序列距离
    return dtw_distance, warping_path
