import pandas as pd
import duckdb
import mlflow
import dagshub
from prefect import flow, task
import os

# Initialize DagsHub + MLflow tracking (make sure DAGSHUB_TOKEN is exported)
dagshub.init(repo_owner='malak.bayoumy41',
             repo_name='python-scripts',
             mlflow=True)

# DuckDB connection string with MotherDuck token (make sure MOTHERDUCK_TOKEN is exported)
MOTHERDUCK_TOKEN = os.getenv("MOTHERDUCK_TOKEN")
DUCKDB_CONN = f"md:titanic_test ?motherduck_token={MOTHERDUCK_TOKEN}"

PREDICTION_TABLE = "predictions"

@task
def extract() -> pd.DataFrame:
    con = duckdb.connect(DUCKDB_CONN)
    query = "SELECT * FROM test;"
    df = con.execute(query).fetchdf()
    return df


@task
def predict(df: pd.DataFrame) -> pd.DataFrame:
    MODEL_NAME = "titanic"
    MODEL_VERSION = "latest" 
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
