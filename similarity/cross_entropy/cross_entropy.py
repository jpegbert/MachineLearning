import numpy as np
import tensorflow as tf

"""
交叉熵python实现
"""


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def cross_entropy_by_python(feature, label):
    """
    纯python实现
    """
    loss1 = -np.sum(label * np.log(softmax(feature)))
    print("loss1: ", loss1)


def cross_entropy_by_tensorflow(feature, label):
    """
    调用tensorflow深度学习框架求解
    """
    sess = tf.Session()
    logits = tf.Variable(feature)
    labels = tf.Variable(label)
    sess.run(tf.global_variables_initializer())
    loss2 = sess.run(tf.losses.softmax_cross_entropy(labels, logits))
    sess.close()
    print("loss2: ", loss2)


if __name__ == '__main__':
    feature = np.asarray([6.5, 4.2, 7.4, 3.5], np.float32)
    label = np.array([1, 0, 0, 0])
    # 纯python实现
    cross_entropy_by_python(feature, label)
    # 调用tensorflow深度学习框架求解
    cross_entropy_by_tensorflow(feature, label)

