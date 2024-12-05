import pandas as pd
import os
import logging

# Configure logging
logging.basicConfig(
    filename="data_processing.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_dataset(file_path):
    """
    Load a dataset from a CSV file.
    """
    try:
        logging.info(f"Loading dataset from {file_path}")
        df = pd.read_csv(file_path)
        logging.info(f"Dataset loaded successfully from {file_path}, Shape: {df.shape}")
        return df
    except FileNotFoundError as e:
        logging.error(f"File not found: {file_path} - {e}")
        raise
    except Exception as e:
        logging.error(f"Error loading dataset from {file_path} - {e}")
        raise

def aggregate_credit_data(credit_df):
    """
    Aggregate credit data to calculate credit-related features.
    """
    try:
        logging.info("Aggregating credit data...")
        credit_features = credit_df.groupby('ID').agg(
            total_paid_in_full=('STATUS', lambda x: (x == 'C').sum()),
            total_on_time=('STATUS', lambda x: (x == '0').sum()),
            total_no_credit=('STATUS', lambda x: (x == 'X').sum()),
            total_late=('STATUS', lambda x: sum(x.isin(['1', '2', '3', '4', '5']))),
            max_delay=('STATUS', lambda x: max([int(i) for i in x if i.isdigit()], default=0))
        ).reset_index()
        logging.info("Credit data aggregated successfully.")
        return credit_features
    except Exception as e:
        logging.error(f"Error aggregating credit data - {e}")
        raise

def classify_client(status):
    """
    Classify clients based on credit status.
    """
    try:
        if status in ['C', 'X', '0']:
            return 'Very Good'
        elif status in ['1']:
            return 'Good'
        elif status in ['2', '3']:
            return 'Not Bad'
        elif status in ['4']:
            return 'Bad'
        elif status in ['5']:
            return 'Very Bad'
        else:
            return 'Unknown'
    except Exception as e:
        logging.error(f"Error classifying client status - {e}")
        raise

def process_data(credit_df, application_df):
  
    try:
        logging.info("Processing data...")

        # Aggregate credit features
        credit_features = aggregate_credit_data(credit_df)
        credit_data = pd.merge(credit_df, credit_features, on='ID', how='inner')

        # Apply classification
        credit_data['STATUS'] = credit_data['STATUS'].apply(classify_client)

        # Add MONTHS_BALANCE for merging
        credit_data['MONTHS_BALANCE'] = credit_df['MONTHS_BALANCE']

        # Merge datasets
        final_df = pd.merge(application_df, credit_data, on='ID', how='inner')

        # Rename columns
        final_df.rename(columns={
            'CODE_GENDER': 'gender',
            'FLAG_OWN_CAR': 'own_car',
            'FLAG_OWN_REALTY': 'own_property',
            'CNT_CHILDREN': 'children',
            'AMT_INCOME_TOTAL': 'income',
            'NAME_INCOME_TYPE': 'income_type',
            'NAME_EDUCATION_TYPE': 'education',
            'NAME_FAMILY_STATUS': 'family_status',
            'NAME_HOUSING_TYPE': 'housing_type',
            'FLAG_MOBIL': 'mobile',
            'FLAG_WORK_PHONE': 'work_phone',
            'FLAG_PHONE': 'phone',
            'FLAG_EMAIL': 'email',
            'CNT_FAM_MEMBERS': 'family_members',
            'MONTHS_BALANCE': 'months_balance',
            'STATUS': 'status',
            'DAYS_BIRTH': 'age_in_days',
            'DAYS_EMPLOYED': 'employment_in_days'
        }, inplace=True)

        # Feature engineering
        final_df['age'] = abs(final_df['age_in_days']) // 365
        final_df['employment_in_days'] = abs(final_df['employment_in_days']) // 365
        final_df.drop(['age_in_days', 'ID'], axis=1, inplace=True)

        logging.info("Data processing completed successfully.")
        return final_df
    except Exception as e:
        logging.error(f"Error processing data - {e}")
        raise

def save_dataset(df, output_folder, file_name):

    try:
        print("*",10)
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, file_name)
        logging.info(f"Saving dataset to {output_path}")
        df.to_csv(output_path, index=False)
        logging.info(f"Dataset saved successfully to {output_path}")
    except Exception as e:
        logging.error(f"Error saving dataset to {output_path} - {e}")
        raise

def main():
  
    print("="*10)

    try:
        # Define paths
        c_path = os.path.join(os.getcwd(), "data/cleaned_data/credits_record.csv")
        a_path = os.path.join(os.getcwd(), "data/cleaned_data/application_record.csv")
        print("="*10,c_path)
        # Load datasets
        credit_df = load_dataset(c_path)
        application_df = load_dataset(a_path)

        # Process data
        final_df = process_data(credit_df, application_df)
        save_dataset(final_df, "data/Featured_data", "featured.csv")

        logging.info("Pipeline executed successfully.")
    except Exception as e:
        logging.error(f"Pipeline execution failed - {e}")
        raise

if __name__ == "__main__":
    print("="*10)
    main()
