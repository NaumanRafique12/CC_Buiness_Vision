import pandas as pd
import numpy as np
import os
import json
import logging
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report
import yaml
max_depth = yaml.safe_load(open("params.yaml"))['model_training']['max_depth']
# Configure logging
logging.basicConfig(
    filename="model_training.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_dataset(file_path):
    """
    Load dataset from a CSV file with exception handling.
    """
    try:
        logging.info(f"Loading dataset from {file_path}")
        df = pd.read_csv(file_path)
        logging.info(f"Dataset loaded successfully, Shape: {df.shape}")
        return df
    except FileNotFoundError as e:
        logging.error(f"File not found: {file_path} - {e}")
        raise
    except Exception as e:
        logging.error(f"Error loading dataset from {file_path} - {e}")
        raise

def define_preprocessor(categorical_columns, numerical_columns):
    """
    Define a preprocessing pipeline for categorical and numerical columns.
    """
    try:
        logging.info("Defining the preprocessing pipeline")
        ohe = OneHotEncoder(handle_unknown='ignore')  # For categorical columns
        scaler = StandardScaler()  # For numerical columns
        oe = OrdinalEncoder(categories=[
            ['Lower secondary', 'Incomplete higher', "Secondary / secondary special", 'Higher education', 'Academic degree']
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', ohe, categorical_columns),  # Apply OHE to categorical columns
                ('num', scaler, numerical_columns),  # Apply StandardScaler to numerical columns
                ('oe', oe, ['education'])  # Apply OrdinalEncoder to 'education'
            ]
        )
        logging.info("Preprocessor defined successfully")
        return preprocessor
    except Exception as e:
        logging.error(f"Error defining preprocessor - {e}")
        raise

def train_and_evaluate_model(X_train, y_train, X_test, y_test, preprocessor):
    """
    Train and evaluate the model using a pipeline.
    """
    try:
        models = {
            'Random Forest': RandomForestClassifier(random_state=42,max_depth=max_depth),
            1:1
        }

        for model_name, model in models.items():
            logging.info(f"Training and evaluating model: {model_name}")
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])

            # Train the pipeline
            pipeline.fit(X_train, y_train)

            # Evaluate the pipeline
            y_pred = pipeline.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted', zero_division=1)
            recall = recall_score(y_test, y_pred, average='weighted', zero_division=1)
            auc = roc_auc_score(y_test, pipeline.predict_proba(X_test), multi_class='ovr')

            logging.info(f"Model: {model_name}, Accuracy: {accuracy:.2f}, Precision: {precision:.2f}, Recall: {recall:.2f}, AUC: {auc:.2f}")
            print(f"{model_name} Accuracy: {accuracy:.2f}")
            print(classification_report(y_test, y_pred))
            # Define file path for saving the pipeline
            import pickle
            import os
            file_path = os.path.join(os.getcwd(),"models",'pipeline.pkl')

 
            with open(file_path, 'wb') as file:
                pickle.dump(pipeline, file)
            print(f"Pipeline saved successfully at {file_path}")
 
            file_path = os.path.join(os.getcwd(),"models",'features.pkl')

            # Save the pipeline
            with open(file_path, 'wb') as file:
                pickle.dump(X_train.columns.to_list(), file)
            print(f"Features saved successfully at {file_path}")
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'auc': auc
            }
    except Exception as e:
        logging.error(f"Error training and evaluating model - {e}")
        raise

def save_metrics(metrics_dict, output_folder):
    """
    Save evaluation metrics to a JSON file.
    """
    try:
        os.makedirs(output_folder, exist_ok=True)
        output_file = os.path.join(output_folder, 'metrics.json')
        logging.info(f"Saving metrics to {output_file}")
        with open(output_file, 'w') as file:
            json.dump(metrics_dict, file, indent=4)
        logging.info("Metrics saved successfully")
    except Exception as e:
        logging.error(f"Error saving metrics - {e}")
        raise

def main():
    """
    Main function to execute the training pipeline.
    """
    try:
        # Load datasets
        logging.info("Starting the model training pipeline")
        tr_csv = os.path.join(os.getcwd(), "Split_Data/train/training.csv")
        ts_csv = os.path.join(os.getcwd(), "Split_Data/test/testing.csv")
        train_df = load_dataset(tr_csv)
        test_df = load_dataset(ts_csv)

        # Split features and target
        X_train, y_train = train_df.drop('status', axis=1), train_df['status']
        X_test, y_test = test_df.drop('status', axis=1), test_df['status']

        # Define columns
        categorical_columns = X_train.select_dtypes(include='object').columns.tolist()
        numerical_columns = X_train.select_dtypes(exclude='object').columns.tolist()
        # ['gender', 'own_car', 'own_property', 'income_type',
        #  'family_status', 'housing_type', 'OCCUPATION_TYPE']
        # ['income', 'employment_in_days', 'age', 'total_paid_in_full',
        #  'total_on_time', 'total_no_credit', 'total_late', 'max_delay',
        #  'work_phone', 'phone', 'email', 'family_members', 'months_balance']

        # Define preprocessor
        preprocessor = define_preprocessor(categorical_columns, numerical_columns)

        # Train and evaluate the model
        metrics = train_and_evaluate_model(X_train, y_train, X_test, y_test, preprocessor)

        # Save metrics to a JSON file
        save_metrics(metrics, "Metrics")

        logging.info("Model training pipeline executed successfully")
    except Exception as e:
        logging.error(f"Pipeline execution failed - {e}")
        raise

if __name__ == "__main__":
    main()
