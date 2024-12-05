import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import logging
import yaml
test_size = yaml.safe_load(open("params.yaml","r"))['data_splitting']['test_size']
# Configure logging
logging.basicConfig(
    filename="data_splitting.log",
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

def encode_labels(df, column):
    """
    Apply label encoding to a specified column.
    """
    try:
        logging.info(f"Applying Label Encoding to column: {column}")
        df[column] = LabelEncoder().fit_transform(df[column])
        logging.info(f"Label Encoding applied successfully to column: {column}")
        return df
    except Exception as e:
        logging.error(f"Error applying Label Encoding to column: {column} - {e}")
        raise

def split_data(df, test_size=0.2, random_state=42):
    """
    Split the dataset into training and testing sets.
    """
    try:
        logging.info(f"Splitting dataset with test size: {test_size}")
        training_data, testing_data = train_test_split(df, test_size=test_size, random_state=random_state)
        logging.info(f"Data split successfully. Training shape: {training_data.shape}, Testing shape: {testing_data.shape}")
        return training_data, testing_data
    except Exception as e:
        logging.error(f"Error splitting data - {e}")
        raise

def save_datasets(training_data, testing_data, main_folder):
    """
    Save training and testing datasets to specified folders.
    """
    try:
        # Define subfolder paths
        train_folder = os.path.join(main_folder, "train")
        test_folder = os.path.join(main_folder, "test")

        # Create folders
        os.makedirs(train_folder, exist_ok=True)
        os.makedirs(test_folder, exist_ok=True)
        logging.info(f"Created folders: {train_folder} and {test_folder}")

        # Save datasets
        training_data.to_csv(os.path.join(train_folder, "training.csv"), index=False)
        testing_data.to_csv(os.path.join(test_folder, "testing.csv"), index=False)
        logging.info(f"Training and testing datasets saved successfully.")
    except Exception as e:
        logging.error(f"Error saving datasets - {e}")
        raise

def main():
    """
    Main function to execute the data splitting pipeline.
    """
    try:

        file_path = os.path.join(os.getcwd(), "data/Final", "Final.csv")

        final_df = load_dataset(file_path)

        final_df = encode_labels(final_df, 'status')
        training_data, testing_data = split_data(final_df)
        save_datasets(training_data, testing_data, "data/Split_Data")

        logging.info("Data splitting pipeline executed successfully.")
    except Exception as e:
        logging.error(f"Pipeline execution failed - {e}")
        raise

if __name__ == "__main__":
    main()
