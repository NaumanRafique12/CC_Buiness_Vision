import optuna
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
import mlflow
import mlflow.sklearn
import os

# Set MLflow tracking URI
mlflow.set_tracking_uri("http://localhost:5001")  # Replace with your MLflow server URI
mlflow.set_experiment("RandomForest_Optuna_Hyperparameter_Tuning")

# Load Dataset
def load_dataset(train_path, test_path, target_column):
    """
    Load and split the dataset.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    X_train, y_train = train_df.drop(target_column, axis=1), train_df[target_column]
    X_test, y_test = test_df.drop(target_column, axis=1), test_df[target_column]
    return X_train, y_train, X_test, y_test

# Preprocessor Definition
def define_preprocessor(categorical_columns, numerical_columns):
    """
    Define preprocessing pipeline.
    """
    ohe = OneHotEncoder(handle_unknown="ignore")
    scaler = StandardScaler()
    oe = OrdinalEncoder(categories=[['Lower secondary', 'Incomplete higher', "Secondary / secondary special", 'Higher education', 'Academic degree']])
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", ohe, categorical_columns),
            ("num", scaler, numerical_columns),
            ("oe", oe, ["education"])  # Example: Ordinal encoding for 'education'
        ]
    )
    return preprocessor

# Objective Function for Optuna
def objective(trial, X_train, y_train, preprocessor, class_weights):
    """
    Objective function for Optuna to optimize Random Forest hyperparameters.
    """
    # Hyperparameter search space
    n_estimators = trial.suggest_int("n_estimators", 50, 300)
    max_depth = trial.suggest_int("max_depth", 3, 20)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 5)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", None])

    # Define the Random Forest model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=42,
        class_weight=class_weights,
    )

    # Build pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # Evaluate with cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=3, scoring="accuracy", n_jobs=-1)
    return np.mean(cv_scores)

# Train the model and log to MLflow
def train_and_log_best_model(X_train, y_train, X_test, y_test, preprocessor, best_params, class_weights):
    """
    Train the best model with tuned hyperparameters and log to MLflow.
    """
    # Define the model with the best hyperparameters
    model = RandomForestClassifier(
        **best_params,
        random_state=42,
       
    )

    # Build pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # Train the pipeline
    pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1] if len(np.unique(y_train)) == 2 else None

    # Metrics
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=1)
    per_class_metrics = {}
    for class_label, metrics in report.items():
        if class_label not in ['accuracy', 'macro avg', 'weighted avg']:  # Skip aggregate metrics
            per_class_metrics[f"precision_class_{class_label}"] = metrics["precision"]
            per_class_metrics[f"recall_class_{class_label}"] = metrics["recall"]
            per_class_metrics[f"f1_score_class_{class_label}"] = metrics["f1-score"]

    # Aggregate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted', zero_division=1),
        "recall": recall_score(y_test, y_pred, average='weighted', zero_division=1),
        "roc_auc": roc_auc_score(y_test, y_proba, multi_class="ovr") if y_proba is not None else None
    }

    # Add per-class metrics to overall metrics
    metrics.update(per_class_metrics)

    # Log metrics and parameters to MLflow
    with mlflow.start_run():
        mlflow.log_params(best_params)
        for key, value in metrics.items():
            if value is not None:
                mlflow.log_metric(key, value)
        mlflow.sklearn.log_model(pipeline, "model")

    print(f"Metrics logged: {metrics}")
    return pipeline

def main():
    # Load dataset
    train_file_path = os.path.join(os.getcwd(), "data/Split_Data/train/training.csv")
    test_file_path = os.path.join(os.getcwd(), "data/Split_Data/test/testing.csv")
    target_column = "status" 

    X_train, y_train, X_test, y_test = load_dataset(train_file_path, test_file_path, target_column)

    categorical_columns = X_train.select_dtypes(include="object").columns.to_list()
    numerical_columns = X_train.select_dtypes(exclude="object").columns.to_list()
    preprocessor = define_preprocessor(categorical_columns, numerical_columns)

    # Compute class weights
    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weights = dict(zip(np.unique(y_train), class_weights))

    # Optimize hyperparameters with Optuna
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: objective(trial, X_train, y_train, preprocessor, class_weights), n_trials=50)

    # Get best hyperparameters
    best_params = study.best_params
    print("Best Parameters:", best_params)
    train_and_log_best_model(X_train, y_train, X_test, y_test, preprocessor, best_params, class_weights)

if __name__ == "__main__":
    main()
