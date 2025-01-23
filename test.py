import mlflow
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)
logged_model = 'runs:/4fcd964815a14991ae81fd3ee3865d26/rf-model'
loded_model = mlflow.pyfunc.load_model(logged_model)
x_predict = loded_model.predict(pd.DataFrame(X_test))
print("x_predict", x_predict)