import pickle

from sklearn import datasets
from sklearn.model_selection import train_test_split

# 使用鸢尾数据集
# Load the Iris dataset
X, y = datasets.load_iris(return_X_y=True)
# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 使用自己收集的数据
# X_test = [[5.1, 3.5, 1.4, 0.2], [7, 3.2, 4.7, 1.4], [5.8, 2.7, 5.1, 1.9], [5, 3.4, 1.5, 0.2]]

# 加载模型文件
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# 预测类别
predict_res = model.predict(X_test)
print(predict_res)
# 类别的概率值
predict_proba_res = model.predict_proba(X_test)
print(predict_proba_res)
