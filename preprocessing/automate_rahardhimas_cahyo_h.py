import pandas as pd
import numpy as np
import glob
import warnings
import os

from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocess_data(input_file, output_file):

    # Load data
    df = pd.read_csv(input_file)

    # Drop kolom
    df.drop("customerID", axis=1, inplace=True)

    # Ubah tipe data
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Missing value
    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["TotalCharges"].median()
    )

    # Encoding
    encoder = LabelEncoder()

    for col in df.select_dtypes(include="object").columns:
        df[col] = encoder.fit_transform(df[col])

    # Scaling
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    processed_df = pd.DataFrame(
        X_scaled,
        columns=X.columns
    )

    processed_df["Churn"] = y

    processed_df.to_csv(
        output_file,
        index=False
    )

    print("Preprocessing selesai!")

if __name__ == "__main__":

    preprocess_data(
        "dataset_raw/WA_Fn-UseC_-Telco-Customer-Churn_raw.csv",
        "preprocessing/WA_Fn-UseC_-Telco-Customer-Churn_preprocessing.csv"
    )