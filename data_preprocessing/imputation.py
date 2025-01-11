from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pandas as pd

# Function to label encode the categorical variables (similar to MissForest handling of categorical data)
def label_encode_with_nan(dataframe):
    label_encoders = {}
    for column in dataframe.select_dtypes(include=['object', 'category']).columns:
        le = LabelEncoder()
        non_null_values = dataframe[column].dropna().unique()
        le.fit(non_null_values)
        label_encoders[column] = le
        dataframe[column] = dataframe[column].apply(lambda x: le.transform([x])[0] if pd.notnull(x) else x)
    return label_encoders

# Function to decode the categorical variables back to their original form
def label_decoders(dataframe, encoders):
    for key, value in encoders.items():
        mapping_encoders = {i: value for i, value in enumerate(value.classes_)}
        dataframe[key] = dataframe[key].replace(mapping_encoders)
    return dataframe

# Function for performing imputation using IterativeImputer with RandomForestRegressor
def imputing_data(dataframe, target_col):
    # Drop the target column to avoid imputing it
    columns_to_impute = dataframe.drop(columns=[target_col])

    # Initialize IterativeImputer with RandomForestRegressor (similar to MissForest)
    imputer = IterativeImputer(estimator=RandomForestRegressor(), random_state=42)

    # Perform imputation
    imputed_data = imputer.fit_transform(columns_to_impute)

    # Convert back to DataFrame
    imputed_df = pd.DataFrame(imputed_data, columns=columns_to_impute.columns)

    # Add the target column back to the DataFrame
    imputed_df[target_col] = dataframe[target_col].values

    return imputed_df

# Function to handle the whole imputation process
def imputation(df, target_col):
    # Label encode the categorical variables
    encoders = label_encode_with_nan(df)

    # Impute the data using IterativeImputer with RandomForest
    imputed_df = imputing_data(df, target_col)

    # Decode the categorical variables back to their original form
    df = label_decoders(imputed_df, encoders)

    return df