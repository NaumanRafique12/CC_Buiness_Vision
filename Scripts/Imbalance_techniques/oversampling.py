import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, roc_auc_score
from imblearn.over_sampling import SMOTE, ADASYN, SVMSMOTE, SMOTENC
from sklearn.utils.class_weight import compute_class_weight
import logging
from imblearn.pipeline import Pipeline as ImbPipeline  # Use imbalanced-learn's pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline 
import mlflow.sklearn
import os
import pickle
from random import random

mlflow.set_tracking_uri("http://localhost:5001")  # Replace with your MLflow server URI

# Configure MLflow experiment
mlflow.set_experiment("Oversampling_RF_Experiment")

def load_dataset(tr_file_path, test_file_path):
    """
    Load dataset from a CSV file.
    """
    try:
        train_df = pd.read_csv(tr_file_path)
        test_df = pd.read_csv(test_file_path)
        train_df = train_df.head(100000)
        test_df = test_df.head(100000)
        X_train, y_train = train_df.drop('status', axis=1), train_df['status']
        X_test, y_test = test_df.drop('status', axis=1), test_df['status']
        return X_train, y_train, X_test, y_test
    except Exception as e:
        print(f"Error loading dataset: {e}")
        raise

def apply_oversampling(technique,categorical_indices=None):
    """
    Apply the specified oversampling technique.
    """
    if technique == "smote":
        sampler = SMOTE(random_state=42)
    elif technique == "adasyn":
        sampler = ADASYN(random_state=42)
    elif technique == "svmsmote":
        sampler = SVMSMOTE(random_state=42)
    elif technique == "smotenc":
        sampler = SMOTENC(random_state=42,categorical_features=categorical_indices)
        
    else:
        raise ValueError(f"Unknown oversampling technique: {technique}")
    
    return sampler
    

def compute_class_weights(y):

    classes = np.unique(y)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
    return dict(zip(classes, weights))

def train_model(X_train, y_train, X_test, y_test, preprocessor, technique, categorical_indices=None, class_weights=None):

    # Define model
    if technique=="class_weights":
        class_weights = compute_class_weights(y_train)

    # Define the RandomForestClassifier with class weights
        model = RandomForestClassifier(random_state=42, class_weight=class_weights)
        pipeline = ImbPipeline(steps=[
        ('preprocessor', preprocessor),
        # ('imbalance', apply_oversampling(technique, categorical_indices)),
        ('classifier', model)
            ])
    else:
        model = RandomForestClassifier(random_state=42, class_weight=class_weights)

    # Build the pipeline
        pipeline = ImbPipeline(steps=[
            ('preprocessor', preprocessor),
            ('imbalance', apply_oversampling(technique, categorical_indices)),
            ('classifier', model)
        ])

    # Train the pipeline
    pipeline.fit(X_train, y_train)

    # Predictions
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1] if len(np.unique(y_train)) == 2 else None

    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=1)

    # Extract precision, recall, and F1-score for each class
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

    return pipeline, metrics, y_pred

def define_preprocessor(categorical_columns, numerical_columns):

    try:
        logging.info("Defining the preprocessing pipeline")
        ohe = OneHotEncoder(handle_unknown='ignore')  # For categorical columns
        scaler = StandardScaler()  # For numerical columns
        oe = OrdinalEncoder(categories=[
            ['Lower secondary', 'Incomplete higher', "Secondary / secondary special", 'Higher education', 'Academic degree']
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', ohe, categorical_columns),  
                ('num', scaler, numerical_columns),
                ('oe', oe, ['education'])  
            ]
        )
        logging.info("Preprocessor defined successfully")
        return preprocessor
    except Exception as e:
        logging.error(f"Error defining preprocessor - {e}")
        raise
def save_artifacts(metrics, technique, run_id):
    """
    Save metrics and other artifacts to MLflow.
    """
    for key, value in metrics.items():
        if value is not None:
            mlflow.log_metric(key, value)
    mlflow.log_param("technique", technique)
    print(f"Metrics logged successfully for technique: {technique}, Run ID: {run_id}")

def main():
    # Load dataset
    train_file_path = os.path.join(os.getcwd(), "data/Split_Data/train/training.csv")
    test_file_path = os.path.join(os.getcwd(), "data/Split_Data/test/testing.csv")
    target_column = "status" 
    
    X_train, y_train, X_test, y_test = load_dataset(train_file_path, test_file_path)
    categorical_columns = X_train.select_dtypes(include='object').columns.to_list()
    numerical_columns = X_train.select_dtypes(exclude='object').columns.to_list()
    categorical_indices = [X_train.columns.get_loc(col) for col in categorical_columns] if categorical_columns else None
    preprocessor = define_preprocessor(categorical_columns, numerical_columns)
    # Techniques to test
    techniques = ['class_weights',"smote", "adasyn", "svmsmote", "smotenc"]

    for technique in techniques:
        with mlflow.start_run() as run:
            print(f"Running {technique.upper()}...")

            if technique == "class_weights":
                model, metrics, y_pred = train_model(X_train, y_train, X_test, y_test,preprocessor,technique,categorical_indices)
            else:
                if technique == "smotenc":
                    categorical_columns  = X_train.select_dtypes(include='object').columns.to_list()
                    categorical_indices = [X_train.columns.get_loc(col) for col in categorical_columns] if categorical_columns else None
                    model, metrics, y_pred = train_model(X_train, y_train, X_test, y_test,preprocessor,technique,categorical_indices)
                else:
                    model, metrics, y_pred = train_model(X_train, y_train, X_test, y_test,preprocessor,technique,categorical_indices)

            # Save metrics and log artifacts
            save_artifacts(metrics, technique, run.info.run_id)


            print(f"Completed {technique.upper()}.")
            

if __name__ == "__main__":
    main()
