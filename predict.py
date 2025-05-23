import mlflow.pyfunc
import pandas as pd
import dagshub
import mlflow

# Initialize Dagshub tracking
dagshub.init(repo_owner='malak.bayoumy41',
             repo_name='python-scripts',
             mlflow=True)

# Load the model from the Production stage
model_uri = "models:/titanic/latest"
model = mlflow.pyfunc.load_model(model_uri)

# Sample input
sample_input = pd.DataFrame({
    "Pclass": [3],
    "Sex": ["male"],
    "SibSp": [1],
    "Parch": [0],
    "Age": [22.0],
    "Fare": [7.25],
    "Embarked": ["S"]
})

# Predict
predictions = model.predict(sample_input)
print("The prediction is",predictions)
