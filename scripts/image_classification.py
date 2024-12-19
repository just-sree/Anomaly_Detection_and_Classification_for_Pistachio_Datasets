import os
import numpy as np
import pandas as pd
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load and preprocess image data
def load_image_dataset(image_folder, image_size=(64, 64)):
    """
    Load images and labels from a folder.

    Parameters:
    - image_folder (str): Path to the folder containing image subfolders for each class.
    - image_size (tuple): Target size for resizing images.

    Returns:
    - X (ndarray): Array of flattened image data.
    - y (ndarray): Array of class labels.
    """
    X, y = [], []
    for label in os.listdir(image_folder):
        class_folder = os.path.join(image_folder, label)
        if not os.path.isdir(class_folder):
            continue
        for image_file in os.listdir(class_folder):
            image_path = os.path.join(class_folder, image_file)
            image = imread(image_path)
            image_resized = resize(image, image_size, anti_aliasing=True, mode='reflect')
            X.append(image_resized.flatten())
            y.append(label)

    print("Images Loaded Successfully.")
    return np.array(X), np.array(y)

# Function to standardize image data
def standardize_images(X):
    """
    Standardize image data to have zero mean and unit variance.

    Parameters:
    - X (ndarray): Image data.

    Returns:
    - X_scaled (ndarray): Standardized image data.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print("Image Data Standardized Successfully.")
    return X_scaled

# Function to train and evaluate image classifiers
def train_image_classifiers(X, y, output_folder):
    """
    Train and evaluate classifiers on image data.

    Parameters:
    - X (ndarray): Image feature data.
    - y (ndarray): Class labels.
    - output_folder (str): Directory to save evaluation results.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    # Define classifiers and hyperparameters
    classifiers = {
        "MLP Classifier": {
            "model": MLPClassifier(max_iter=1000, random_state=42),
            "params": {
                "hidden_layer_sizes": [(50,), (100,), (50, 50)],
                "alpha": [0.0001, 0.001, 0.01],
                "learning_rate_init": [0.001, 0.01]
            }
        },
        "Naive Bayes": {
            "model": GaussianNB(),
            "params": {
                "var_smoothing": np.logspace(-10, -7, 4)
            }
        }
    }

    results = []

    for classifier_name, classifier in classifiers.items():
        print(f"Training {classifier_name} on Image Dataset")
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

        print(f"{classifier_name} Results:")
        print(f"Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        print(classification_report(y_test, predictions))

        results.append({
            "Classifier": classifier_name,
            "Accuracy": accuracy,
            "F1 Score": f1,
            "Precision": precision,
            "Recall": recall,
            "Best Params": random_search.best_params_
        })

    # Save results
    results_df = pd.DataFrame(results)
    results_file = os.path.join(output_folder, "image_classification_results.csv")
    results_df.to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")

    # Visualize results
    visualize_image_results(results_df, output_folder)

# Function to visualize classification results
def visualize_image_results(results_df, output_folder):
    """
    Visualize classification metrics for image data.

    Parameters:
    - results_df (DataFrame): DataFrame containing classification results.
    - output_folder (str): Directory to save plots.
    """
    metrics = ["Accuracy", "F1 Score", "Precision", "Recall"]

    for metric in metrics:
        plt.figure(figsize=(10, 7))
        sns.barplot(x=metric, y="Classifier", data=results_df, palette="coolwarm")
        plt.title(f"Image Classifier {metric} Comparison")
        plt.tight_layout()
        plot_path = os.path.join(output_folder, f"image_classifier_{metric.lower()}_comparison.png")
        plt.savefig(plot_path)
        plt.show()

# Main execution workflow
if __name__ == "__main__":
    # Example image dataset folder
    image_folder = "Pistachio_Images"
    output_folder = "image_classification_outputs"

    # Load and preprocess data
    X, y = load_image_dataset(image_folder)
    X_scaled = standardize_images(X)

    # Train and evaluate classifiers
    train_image_classifiers(X_scaled, y, output_folder)

    print("Image Classification Completed Successfully.")
