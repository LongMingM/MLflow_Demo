import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# 第一部分：数据集建模
# 加载并准备用于建模的 Iris 数据集。
# 训练逻辑回归模型并评估其性能。
# 准备模型超参数并计算日志记录指标。
# Load the iris dataset
X, y = datasets.load_iris(return_X_y=True)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2, random_state=42)

#define the model hyperparameters
params = {
    "solver": "lbfgs",
    "max_iter": 1000,
    "multi_class": "auto",
    "random_state": 8888,
}

#train the model
lr = LogisticRegression(**params)
lr.fit(X_train, y_train)

# Predict the labels for the test set
y_pred = lr.predict(X_test)

#calculate the metrics
accuracy = accuracy_score(y_test, y_pred)



# 第二部分：关联
# 如果使用的是 Databricks 未提供的托管 MLflow 跟踪服务器，或者运行本地跟踪服务器，请确保使用以下命令设置跟踪服务器的 URI：
# Log the model and metrics to MLflow
mlflow.set_tracking_uri("http://localhost:8080/")

# create a new experiment
mlflow.set_experiment("Iris_Classification")



# 第三部分：启动
# 启动 MLflow 运行上下文以启动新运行，我们将模型和元数据记录到该运行。
# 记录模型参数和性能指标。
# 标记运行以便于检索。
# 在记录（保存）模型时，在 MLflow 模型注册表中注册模型。
with mlflow.start_run():
    # Log model parameters
    mlflow.log_params(params)

    # Log the loss metrics
    mlflow.log_metric("accuracy", accuracy)

    # Set a tag that we can use to remind ourselves what this run was about
    mlflow.set_tag("Training Info", "Basic LR model for Iris dataset")

    # Infer the model signature
    signature = infer_signature(X_train, lr.predict(X_train))

    #log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=lr,
        artifact_path="iris_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="Iris_Classification_Model"
    )

# Load the model back for predictions as a generic Python Function model
loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

predictions = loaded_model.predict(X_test)

iris_feature_names = datasets.load_iris().feature_names

result = pd.DataFrame(X_test, columns=iris_feature_names)
result["actual_class"] = y_test
result["predicted_class"] = predictions


result[:4]

