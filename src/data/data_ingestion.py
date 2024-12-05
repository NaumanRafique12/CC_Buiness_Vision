import pandas as pd
import os

# Define paths for the input datasets
c_path = os.path.join(os.getcwd(),"data/archive/credit_record.csv")
a_path = os.path.join(os.getcwd(),"data/archive/application_record.csv")
print(os.getcwd())
# Load the datasets
print("="*10,c_path)
credit_df = pd.read_csv(c_path)
application_df = pd.read_csv(a_path)

application_df = application_df.drop_duplicates()
credit_df = credit_df.drop_duplicates()


# Define the path for the new folder to save cleaned datasets
new_folder_path = "data/cleaned_data"
os.makedirs(new_folder_path, exist_ok=True)

application_df.to_csv(os.path.join(new_folder_path, "application_record.csv"), index=False)
credit_df.to_csv(os.path.join(new_folder_path, "credits_record.csv"), index=False)


new_folder_path = "data/cleaned_data"
os.makedirs(new_folder_path, exist_ok=True)

application_df.to_csv(os.path.join(new_folder_path, "application_record.csv"), index=False)
credit_df.to_csv(os.path.join(new_folder_path, "credits_record.csv"), index=False)


