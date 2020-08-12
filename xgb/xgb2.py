import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

'''
https://github.com/HuangQinJian/ML_Code/blob/master/XGBoost/XGBoost_2.py
基于XGBoost原生接口的回归
'''

# 读取文件原始数据
data = []
labels = []
labels2 = []
with open("lppz5.csv", encoding='UTF-8') as fileObject:
    for line in fileObject:
        line_split = line.split(',')
        data.append(line_split[10:])
        labels.append(line_split[8])

X = []
for row in data:
    row = [float(x) for x in row]
    X.append(row)

y = [float(x) for x in labels]

# XGBoost训练过程
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

params = {
    'booster': 'gbtree',
    'objective': 'reg:gamma',
    'gamma': 0.1,
    'max_depth': 5,
    'lambda': 3,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'min_child_weight': 3,
    'silent': 1,
    'eta': 0.1,
    'seed': 1000,
    'nthread': 4,
}

dtrain = xgb.DMatrix(X_train, y_train)
num_rounds = 300
plst = params.items()
model = xgb.train(plst, dtrain, num_rounds)

# 对测试集进行预测
dtest = xgb.DMatrix(X_test)
ans = model.predict(dtest)

# 显示重要特征
plot_importance(model)
plt.show()
