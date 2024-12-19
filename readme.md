Anomaly Detection and Classification for Pistachio Datasets

## Overview
This project focuses on anomaly detection and classification using machine learning techniques. The workflow includes preprocessing datasets, detecting outliers, performing dimensionality reduction, and building classification models. Additionally, the project applies these techniques to image data for pistachio classification. The analysis is based on a comprehensive pistachio dataset and follows the CRISP-DM methodology for a structured approach.

### Key Features:
- **Outlier Detection:** Implements LOF, Isolation Forest, and One-Class SVM for anomaly detection.
- **Classification Models:** Evaluates Random Forest, Naive Bayes, and Multi-Layer Perceptron (MLP) classifiers.
- **Image Classification:** Applies feature engineering and model tuning for pistachio dataset.
- **Dimensionality Reduction:** Employs PCA and LDA to enhance classification performance.
- **Visualization:** Provides insights through detailed plots and performance metrics.

---

## Datasets
The project utilizes the following datasets:

### Pistachio Feature Dataset
- **Classes:** 2
- **Data Type:** Integer, Real
- **Number of Instances:** 2148
- **Number of Features:** 28
- **Year:** 2021
- **Citation Request:**
  - OZKAN IA., KOKLU M., and SARACOGLU R. (2021). Classification of Pistachio Species Using Improved K-NN Classifier. Progress in Nutrition, Vol. 23, N. 2. https://doi.org/10.23751/pn.v23i2.9686. (Open Access)

### Pistachio Image Dataset
- **Classes:** 2
- **Data Type:** Image
- **Number of Instances:** 2148
- **Image Resolution:** Standardized
- **Year:** 2022
- **Citation Requests:**
  1. SINGH D, TASPINAR YS, KURSUN R, CINAR I, KOKLU M, OZKAN IA, LEE H-N. (2022). Classification and Analysis of Pistachio Species with Pre-Trained Deep Learning Models, Electronics, 11 (7), 981. https://doi.org/10.3390/electronics11070981. (Open Access)
  2. OZKAN IA., KOKLU M., and SARACOGLU R. (2021). Classification of Pistachio Species Using Improved K-NN Classifier. Progress in Nutrition, Vol. 23, N. 2. https://doi.org/10.23751/pn.v23i2.9686. (Open Access)

---

## Setup Instructions
Follow the steps below to set up and run the project:

### Prerequisites
- Python 3.8+
- Libraries: Install dependencies listed in `requirements.txt`.

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/AML_Project.git
    cd AML_Project
    ```

2. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

### Running the Code
1. Preprocessing and Outlier Detection:
    ```bash
    python scripts/anomaly_detection.py
    ```

2. Classification:
    ```bash
    python scripts/classification.py
    ```

3. Image Classification:
    ```bash
    python scripts/image_classification.py
    ```

4. View Results:
    - Classification plots and outputs are saved in `outputs/`.

---

## Key Results
### Outlier Detection
- Detected common outliers across datasets (e.g., 11 in corrected data, 22 in PCA-reduced data).
- LDA-reduced data showed minimal outliers, highlighting its effectiveness in class separation.

### Classification Performance
- **Random Forest:** Achieved highest accuracy (99.53%) on PCA-reduced data.
- **Naive Bayes:** Performed best on PCA-reduced data with 99.30% accuracy.
- **MLP:** Perfect classification (100% accuracy) on PCA-reduced data.

### Image Classification
- MLPClassifier: Achieved 87.67% accuracy, outperforming GaussianNB.
- GaussianNB: Lower accuracy at 60.70%, indicating limitations for image data.

---

 Project Structure
```
Anomaly_Detection_and_Classification_for_Pistachio_Datasets/
├── data/                Placeholder for datasets
├── notebooks/           Jupyter Notebooks for interactive exploration
├── outputs/             Saved results and plots
├── scripts/             Modular Python scripts
├── README.md            Project overview
├── requirements.txt     Dependencies
└── .gitignore           Ignored files
```
---

 Contact
For any queries, feel free to reach out:
- Name: Sree Sankaran Chackoth
- Email: sreechackoth@gmail.com
- GitHub: [just-sree](https://github.com/just-sree)

---