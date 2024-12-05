import pandas as pd
from scipy.stats import f_oneway, chi2_contingency
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import os
import logging
import yaml
correlation_threshold = yaml.safe_load(open("params.yaml","r"))['statistical_features']['correlation_threshold']
# Configure logging
logging.basicConfig(
    filename="feature_selection.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def load_dataset(file_path):
    """
    Load dataset with exception handling.
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

def find_significant_features(df, target_column, numerical_features, categorical_features, alpha=0.05):
    """
    Identify significant features based on statistical tests.
    """
    try:
        significant_numerical = []
        significant_categorical = []
        target_classes = df[target_column].nunique()

        # Process numerical features
        for feature in numerical_features:
            logging.info(f"Processing numerical feature: {feature}")
            if target_classes == 2:
                # One-Way ANOVA for binary targets
                groups = [df[df[target_column] == cls][feature] for cls in df[target_column].unique()]
                f_stat, p_value = f_oneway(*groups)
            else:
                # Two-Way ANOVA for multi-class targets
                formula = f"{feature} ~ C({target_column})"
                model = ols(formula, data=df).fit()
                anova_results = anova_lm(model)
                p_value = anova_results['PR(>F)'][0]

            if p_value < alpha:
                significant_numerical.append(feature)

        # Process categorical features
        for feature in categorical_features:
            logging.info(f"Processing categorical feature: {feature}")
            contingency_table = pd.crosstab(df[feature], df[target_column])
            chi2_stat, p_value, dof, expected = chi2_contingency(contingency_table)
            if p_value < alpha:
                significant_categorical.append(feature)

        # Manually adding specific features for significance
        significant_categorical.extend(['status', 'months_balance', 'education'])
        significant_numerical.extend(significant_categorical)

        logging.info("Feature selection completed successfully.")
        return significant_numerical
    except Exception as e:
        logging.error(f"Error during feature selection - {e}")
        raise

def drop_highly_correlated_features(df, correlation_threshold=0.8):
    """
    Drop highly correlated features based on a correlation threshold.
    """
    try:
        corr_matrix = df.select_dtypes(exclude='object').corr()
        to_drop = set()

        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > correlation_threshold:
                    colname = corr_matrix.columns[i]
                    to_drop.add(colname)

        df = df.drop(columns=list(to_drop))
        logging.info(f"Dropped highly correlated features: {to_drop}")
        return df
    except Exception as e:
        logging.error(f"Error dropping highly correlated features - {e}")
        raise

def save_dataset(df, output_folder, file_name):

    try:
        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, file_name)
        df.to_csv(output_path, index=False)
        logging.info(f"Dataset saved successfully to {output_path}")
    except Exception as e:
        logging.error(f"Error saving dataset to {output_path} - {e}")
        raise

def main():

    try:
        # Define file path and load dataset
        f_csv = os.path.join(os.getcwd(), "data/Featured_data/featured.csv")
        print(f_csv)
        final_df = load_dataset(f_csv)

        numerical_features = ['income', 'employment_in_days', 'age']
        categorical_features = ['gender', 'own_car', 'OCCUPATION_TYPE', 'family_members', 'children']
        target_column = 'status'
        final_df['OCCUPATION_TYPE'] = final_df['OCCUPATION_TYPE'].fillna('Unknown')

        # Identify significant features
        important_features = find_significant_features(final_df, target_column, numerical_features, categorical_features)
        final_df = final_df[important_features]

        # Drop highly correlated features
        final_df = drop_highly_correlated_features(final_df, correlation_threshold=correlation_threshold)

        # Save the final cleaned and processed dataset
        save_dataset(final_df, "data/Final", "Final.csv")

        logging.info("Feature selection pipeline executed successfully.")
    except Exception as e:
        logging.error(f"Pipeline execution failed - {e}")
        raise

if __name__ == "__main__":
    main()

