import pandas as pd
import duckdb
import mlflow
import dagshub
from prefect import flow, task
import os

# Environment variables already exported — just use them
MOTHERDUCK_TOKEN = os.getenv("MOTHERDUCK_TOKEN")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")
DUCKDB_CONN = f"md:titanic_test ?motherduck_token={MOTHERDUCK_TOKEN}"
MODEL_NAME = "titanic"
MODEL_VERSION = "2"
PREDICTION_TABLE = "predictions"

# Authenticate with DagsHub
dagshub.auth.add_app_token(DAGSHUB_TOKEN)
from mlflow.tracking import MlflowClient

client = MlflowClient()

# Get a specific registered model if you know the name
try:
    model = client.get_registered_model("titanic")
    print("Model found:", model.name)
except Exception as e:
    print("Model not found:", str(e))


@task
def extract() -> pd.DataFrame:
    con = duckdb.connect(DUCKDB_CONN)
    query = "SELECT * FROM test;"
    df = con.execute(query).fetchdf()
    return df


@task
def predict(df: pd.DataFrame) -> pd.DataFrame:
    model_uri = f"models:/{MODEL_NAME}/{MODEL_VERSION}"
    loaded_model = mlflow.pyfunc.load_model(model_uri)

    # Make prediction
    predictions = loaded_model.predict(df)
    df["prediction"] = predictions
    return df


@task
def load(df: pd.DataFrame):
    con = duckdb.connect(DUCKDB_CONN)

    # Ensure predictions table exists
    con.execute(f"CREATE TABLE IF NOT EXISTS {PREDICTION_TABLE} AS SELECT * FROM df LIMIT 0")
    con.register("df", df)

    # Append predictions
    con.execute(f"INSERT INTO {PREDICTION_TABLE} SELECT * FROM df")
    print(f"Predictions written to {PREDICTION_TABLE}")


@flow(name="Titanic Batch Prediction Job")
def titanic_batch_job():
    df = extract()
    preds = predict(df)
    load(preds)


if __name__ == "__main__":
    titanic_batch_job()
