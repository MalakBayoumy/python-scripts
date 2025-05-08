import pandas as pd
from hydra import initialize, compose
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def load_config():
    initialize(config_path=".", version_base=None)
    return compose(config_name="config")


def load_data(cfg):
    train = pd.read_csv(cfg.data.train)
    test = pd.read_csv(cfg.data.test)

    if "Cabin" in train.columns:
        train.drop("Cabin", axis=1, inplace=True)
    if "Cabin" in test.columns:
        test.drop("Cabin", axis=1, inplace=True)

    return train, test


def build_pipeline(cfg, model_name):
    num_transformer = SimpleImputer(strategy=cfg.preprocessing.num_strategy)
    cat_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy=cfg.preprocessing.cat_strategy)),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", num_transformer, list(cfg.preprocessing.num_features)),
        ("cat", cat_transformer, list(cfg.preprocessing.cat_features))
    ])

    if model_name == "RandomForest":
        model = RandomForestClassifier(
            n_estimators=cfg.model.params.n_estimators,
            max_depth=cfg.model.params.max_depth,
            random_state=cfg.model.params.random_state
        )
    elif model_name == "LogisticRegression":
        model = LogisticRegression(max_iter=1000)
    else:
        raise ValueError("Unsupported model type")

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])


def main():
    cfg = load_config()

    train, test = load_data(cfg)
    X = train[cfg.data.features]
    y = train[cfg.data.target_column]
    X_test = test[cfg.data.features]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=cfg.data.test_size,
        random_state=cfg.data.random_state
    )

    pipeline = build_pipeline(cfg, cfg.model.model_name)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_val)

    print(f"\nModel: {cfg.model.model_name}")
    print("Accuracy:", accuracy_score(y_val, y_pred))
    print("Classification Report:\n", classification_report(y_val, y_pred))

    test_preds = pipeline.predict(X_test)
    print("Test predictions complete.")

     # save pipeline to pkl file
    os.makedirs(cfg.model.model_path, exist_ok=True) 
    model_file_path = cfg.model.trained_model_path + ".pkl"
    joblib.dump(pipeline, model_file_path)
    print(f"Model saved to: {model_file_path}")


if __name__ == "__main__":
    main()
