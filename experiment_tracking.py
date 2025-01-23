import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from sklearn.datasets import load_iris
from mlflow.models.signature import infer_signature

import pandas as pd
import joblib

# Load dataset
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=42)

mlflow.set_experiment("Random Forest on Iris dataset")
# Track experiments
random_state=42
for n_estimators in [10, 50, 100]:
    with mlflow.start_run():
        mlflow.log_param("random_state", random_state)

        # Train model
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Prepare input example
        input_example = X_test[:5]  # First 5 rows of test data

        # Infer signature
        signature = infer_signature(X_test, y_pred)

        # Log parameters, metrics, and model
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("random_state", random_state)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("mean_squared_error", mse)
        mlflow.log_metric("r2_score", r2)
        
        #long the model
        mlflow.sklearn.log_model(
            model, 
            "rf-model",
            input_example=input_example,
            signature=signature,
        )

        #dump/serialise the model
        joblib.dump(model, 'models/rf-model')
        #log the artifacts
        mlflow.log_artifact('models/rf-model')
