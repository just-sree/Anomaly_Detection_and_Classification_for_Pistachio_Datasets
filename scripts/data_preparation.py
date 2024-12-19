import os
import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load data
def load_data(file_path):
    """
    Load the dataset into a pandas DataFrame.

    Parameters:
    - file_path (str): Path to the dataset file.

    Returns:
    - df (DataFrame): Loaded dataset.
    """
    df = pd.read_excel(file_path, engine='openpyxl')
    print("Data Loaded Successfully.")
    print(df.head())
    return df

# Function to visualize class distribution
def visualize_class_distribution(df):
    """
    Visualize the distribution of classes in the dataset.
    
    Parameters:
    - df (DataFrame): Dataset with a 'Class' column.
    """
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Class', data=df)
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.grid(True)
    plt.show()

# Function to balance data using SMOTE
def balance_data(df, target_column):
    """
    Balance the dataset using SMOTE.

    Parameters:
    - df (DataFrame): Dataset to balance.
    - target_column (str): Name of the target column.

    Returns:
    - df_resampled (DataFrame): Balanced dataset.
    """
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[target_column])
    X = df.drop(columns=[target_column])

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    df_resampled = pd.DataFrame(X_resampled, columns=X.columns)
    df_resampled[target_column] = label_encoder.inverse_transform(y_resampled)

    print("Data Balanced Successfully.")
    return df_resampled

# Function to standardize data
def standardize_data(df):
    """
    Standardize the dataset.

    Parameters:
    - df (DataFrame): Dataset to standardize.

    Returns:
    - DataFrame: Standardized dataset.
    """
    scaler = StandardScaler()
    df_standardized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    print("Data Standardized Successfully.")
    return df_standardized

# Function to normalize data
def normalize_data(df):
    """
    Normalize the dataset.

    Parameters:
    - df (DataFrame): Dataset to normalize.

    Returns:
    - DataFrame: Normalized dataset.
    """
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    print("Data Normalized Successfully.")
    return df_normalized

# Function for PCA dimensionality reduction
def apply_pca(df, n_components):
    """
    Apply PCA to reduce dimensions of the dataset.

    Parameters:
    - df (DataFrame): Dataset to reduce dimensions.
    - n_components (int): Number of components to keep.

    Returns:
    - DataFrame: Dataset with reduced dimensions.
    """
    pca = PCA(n_components=n_components)
    df_pca = pd.DataFrame(pca.fit_transform(df))
    print("PCA Applied Successfully.")
    return df_pca, pca.explained_variance_ratio_

# Function to plot PCA scree plot
def plot_pca_scree(explained_variance_ratio):
    """
    Plot the scree plot for PCA.

    Parameters:
    - explained_variance_ratio (array): Array of explained variance ratios.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o')
    plt.title('Scree Plot for PCA')
    plt.xlabel('Number of Components')
    plt.ylabel('Explained Variance Ratio')
    plt.grid(True)
    plt.show()

# Function for LDA dimensionality reduction
def apply_lda(df, target, n_components):
    """
    Apply LDA to reduce dimensions of the dataset.

    Parameters:
    - df (DataFrame): Dataset to reduce dimensions.
    - target (Series): Target variable.
    - n_components (int): Number of components to keep.

    Returns:
    - DataFrame: Dataset with reduced dimensions.
    """
    lda = LDA(n_components=n_components)
    df_lda = pd.DataFrame(lda.fit_transform(df, target))
    print("LDA Applied Successfully.")
    return df_lda

# Main execution workflow
if __name__ == "__main__":
    # Load the dataset
    file_path = "Pistachio_28_Features_Dataset.xlsx"
    df = load_data(file_path)

    # Visualize class distribution
    visualize_class_distribution(df)

    # Balance the dataset
    df_balanced = balance_data(df, target_column='Class')

    # Standardize and normalize the data
    df_standardized = standardize_data(df_balanced.drop(columns=['Class']))
    df_normalized = normalize_data(df_balanced.drop(columns=['Class']))

    # Apply PCA
    df_pca, explained_variance_ratio = apply_pca(df_standardized, n_components=10)
    plot_pca_scree(explained_variance_ratio)

    # Apply LDA
    df_lda = apply_lda(df_standardized, df_balanced['Class'], n_components=1)

    print("Data Preparation Completed Successfully.")
