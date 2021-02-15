
"""
兰氏距离python实现
"""


def canberra_distance(p, q):
    n = len(p)
    distance = 0
    for i in n:
        if p[i] == 0 and q[i] == 0:
            distance += 0
        else:
            distance += abs(p[i] - q[i]) / (abs(p[i]) + abs(q[i]))
    return distance

