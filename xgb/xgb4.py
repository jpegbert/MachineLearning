import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

'''
https://github.com/HuangQinJian/ML_Code/blob/master/XGBoost/XGBoost_4.py
基于Scikit-learn接口的回归
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

model = xgb.XGBRegressor(max_depth=5, learning_rate=0.1, n_estimators=160, silent=True, objective='reg:gamma')
model.fit(X_train, y_train)

# 对测试集进行预测
ans = model.predict(X_test)

# 显示重要特征
plot_importance(model)
plt.show()