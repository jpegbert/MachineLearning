from lr.demo1 import lr_model
from sklearn import datasets

"""
LR多分类
"""


def multi_class_classification():
    digits = datasets.load_digits()
    x = digits['data']
    y = digits['target']
    lr_model.logistic_regression(x, y)


multi_class_classification()


