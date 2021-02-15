import numpy as np

"""
Tanimoto系数（广义Jaccard相似系数）python实现
"""


def tanimoto_coefficient(p_vec, q_vec):
    pq = np.dot(p_vec, q_vec)
    p_square = np.linalg.norm(p_vec)
    q_square = np.linalg.norm(q_vec)
    return pq / (p_square + q_square - pq)




