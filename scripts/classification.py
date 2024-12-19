import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

# Function to train and evaluate classifiers
def train_and_evaluate(data_files, output_folder):
    """
    Train and evaluate classifiers on multiple datasets.

    Parameters:
    - data_files (list): List of dataset file paths.
    - output_folder (str): Directory to save evaluation results.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Define classifiers and their hyperparameters
    classifiers = {
        "Random Forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5, 10]
            }
        },
        "Naive Bayes": {
            "model": GaussianNB(),
            "params": {
                "var_smoothing": np.logspace(-10, -7, 4)
            }
        },
        "MLP Classifier": {
            "model": MLPClassifier(max_iter=1000, random_state=42),
            "params": {
                "hidden_layer_sizes": [(50,), (100,), (50, 50)],
                "alpha": [0.0001, 0.001, 0.01],
                "learning_rate_init": [0.001, 0.01]
            }
        }
    }

    results = []

    for file_path in data_files:
        print(f"Processing dataset: {file_path}")
        df = pd.read_csv(file_path)

        # Split dataset into features and target
        X = df.drop(columns=['Class'])
        y = df['Class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        for classifier_name, classifier in classifiers.items():
            print(f"Training {classifier_name} on {file_path}")
            random_search = RandomizedSearchCV(
                estimator=classifier["model"],
                param_distributions=classifier["params"],
                n_iter=10,
                scoring='f1_macro',
                cv=3,
                random_state=42,
                n_jobs=-1
            )
            random_search.fit(X_train, y_train)

            # Evaluate the best model
            best_model = random_search.best_estimator_
            predictions = best_model.predict(X_test)

            accuracy = accuracy_score(y_test, predictions)
            f1 = f1_score(y_test, predictions, average='macro')
            precision = precision_score(y_test, predictions, average='macro')
            recall = recall_score(y_test, predictions, average='macro')

            print(f"{classifier_name} Results on {file_path}:")
            print(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
            print(classification_report(y_test, predictions))

            results.append({
                "Dataset": os.path.basename(file_path),
                "Classifier": classifier_name,
                "Accuracy": accuracy,
                "F1 Score": f1,
                "Precision": precision,
                "Recall": recall,
                "Best Params": random_search.best_params_
            })

    # Save results
    results_df = pd.DataFrame(results)
    results_file = os.path.join(output_folder, "classification_results.csv")
    results_df.to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")

    # Visualize results
    visualize_results(results_df, output_folder)

# Function to visualize classification results
def visualize_results(results_df, output_folder):
    """
    Visualize classification metrics across classifiers and datasets.

    Parameters:
    - results_df (DataFrame): DataFrame containing classification results.
    - output_folder (str): Directory to save plots.
    """
    metrics = ["Accuracy", "F1 Score", "Precision", "Recall"]

    for metric in metrics:
        plt.figure(figsize=(10, 7))
        sns.barplot(x=metric, y="Dataset", hue="Classifier", data=results_df, palette="coolwarm")
        plt.title(f"Classifier {metric} Comparison Across Datasets")
        plt.tight_layout()
        plot_path = os.path.join(output_folder, f"classifier_{metric.lower()}_comparison.png")
        plt.savefig(plot_path)
        plt.show()

# Main execution workflow
if __name__ == "__main__":
    # Example dataset paths
    data_files = [
        "output/corrected_data.csv",
        "output/standardized_data.csv",
        "output/normalized_data.csv",
        "output/pca_reduced_data.csv",
        "output/lda_reduced_data.csv"
    ]

    output_folder = "classification_outputs"
    train_and_evaluate(data_files, output_folder)

    print("Classification Completed Successfully.")
