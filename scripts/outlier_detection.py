import os
import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
import seaborn as sns

# Function to detect anomalies in multiple datasets
def detect_anomalies(data_files, output_folder):
    """
    Detect anomalies using LOF, Isolation Forest, and One-Class SVM.

    Parameters:
    - data_files (list): List of dataset file paths.
    - output_folder (str): Directory to save the outputs with anomaly scores.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_path in data_files:
        print(f"Processing dataset: {file_path}")
        df = pd.read_csv(file_path)

        # Separate features and labels
        X = df.drop(columns=['Class'])

        # Initialize outlier detection models
        models = {
            "LOF": LocalOutlierFactor(n_neighbors=20, contamination=0.1),
            "Isolation Forest": IsolationForest(contamination=0.1, random_state=42),
            "One-Class SVM": OneClassSVM(nu=0.1, gamma='scale')
        }

        # Detect anomalies
        anomaly_results = {}
        for name, model in models.items():
            if name == "LOF":
                # LOF requires fit_predict directly
                anomaly_results[name] = model.fit_predict(X)
            else:
                model.fit(X)
                anomaly_results[name] = model.predict(X)

        # Consolidate results
        df['LOF_Outlier'] = anomaly_results['LOF'] == -1
        df['ISF_Outlier'] = anomaly_results['Isolation Forest'] == -1
        df['OCSVM_Outlier'] = anomaly_results['One-Class SVM'] == -1

        # Save the DataFrame with anomaly scores
        output_file_path = os.path.join(output_folder, os.path.basename(file_path))
        df.to_csv(output_file_path, index=False)

        # Visualization
        visualize_anomalies(X, df, file_path)

# Function to visualize outliers
def visualize_anomalies(X, df, file_name):
    """
    Visualize outliers for each detection method.

    Parameters:
    - X (DataFrame): Feature data.
    - df (DataFrame): Dataset with outlier labels.
    - file_name (str): Name of the dataset file for saving plots.
    """
    plt.figure(figsize=(16, 5))

    methods = ['LOF_Outlier', 'ISF_Outlier', 'OCSVM_Outlier']
    for i, method in enumerate(methods, 1):
        plt.subplot(1, 3, i)
        sns.scatterplot(
            x=X.iloc[:, 0], y=X.iloc[:, 1], 
            hue=df[method], palette={True: 'red', False: 'blue'}, alpha=0.6
        )
        plt.title(f"Outliers by {method.split('_')[0]}")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")

    plt.tight_layout()
    plot_path = f"outlier_visualization_{os.path.basename(file_name)}.png"
    plt.savefig(plot_path)
    plt.show()

# Main execution workflow
if __name__ == "__main__":
    # Example dataset paths
    output_files = [
        "output/corrected_data.csv",
        "output/standardized_data.csv",
        "output/normalized_data.csv",
        "output/pca_reduced_data.csv",
        "output/lda_reduced_data.csv"
    ]

    output_folder = "outlier_detection_outputs"
    detect_anomalies(output_files, output_folder)

    print("Outlier Detection Completed Successfully.")
