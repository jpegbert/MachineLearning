import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.metrics as metric
import statsmodels.api as sm


cancer = load_breast_cancer()
data = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
data['y'] = cancer['target']

# 标准化
scaler = StandardScaler()
scaler.fit(data)
scaled = scaler.transform(data)

# PCA
pca = PCA().fit(scaled)

pc = pca.transform(scaled)
pc1 = pc[:, 0]
pc2 = pc[:, 1]

# 画出主成分
plt.figure(figsize=(10, 10))
colour = ['#ff2121' if y == 1 else '#2176ff' for y in data['y']]
plt.scatter(pc1, pc2, c=colour, edgecolors='#000000')
plt.ylabel("Glucose", size=20)
plt.xlabel('Age', size=20)
plt.yticks(size=12)
plt.xticks(size=12)
plt.xlabel('PC1')

# 绘制PCA-scree图
# percentage of variance explained
var = pca.explained_variance_[0:10]
labels = ['PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10']
plt.figure(figsize=(15, 7))
plt.bar(labels, var,)
plt.xlabel('Pricipal Component')
plt.ylabel('Proportion of Variance Explained')

# PCA-特征组 对比
# 第一组包含所有与对称性和光滑性有关的特征
group_1 = ['mean symmetry', 'symmetry error', 'worst symmetry', 'mean smoothness', 'smoothness error', 'worst smoothness']
# 第二组包含所有与周长和凹陷性有关的特征
group_2 = ['mean perimeter', 'perimeter error', 'worst perimeter', 'mean concavity', 'concavity error', 'worst concavity']
# 这里可以使用两组特征分别按照上面的流程做pca


def logist_groupdata(group):
    """
    使用两组特征的logistic回归模型
    """
    for i, g in enumerate(group):
        x = data[g]
        x = sm.add_constant(x)
        y = data['y']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)
        model = sm.Logit(y_train, x_train).fit() # fit logistic regression model
        predictions = np.around(model.predict(x_test))
        accuracy = metric.accuracy_score(y_test, predictions)
        print("Accuracy of Group {}: {}".format(i+1, accuracy))


logist_groupdata(group_1)
logist_groupdata(group_2)

