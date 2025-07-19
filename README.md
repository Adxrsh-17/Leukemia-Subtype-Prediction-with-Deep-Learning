Deep Learning-Based Leukemia Subtype Classification Using Gene Expression Data
Overview
This project builds a machine learning and deep learning pipeline to classify leukemia subtypes — specifically differentiating between Acute Lymphoblastic Leukemia (ALL) and Acute Myeloid Leukemia (AML) — using high-dimensional gene expression data. The workflow incorporates thorough preprocessing, dimensionality reduction, model training, and result visualization, suitable for reproducible research or application in biomedical informatics.

Table of Contents
Project Objective

Dataset Description

Environment & Dependencies

Pipeline & Methodology

Key Results

Visualizations

How to Run

References

Project Objective
Goal: Predict and classify leukemia subtypes (ALL vs. AML) from microarray-based gene expression profiles.

Approach: Employ scalable preprocessing, dimensionality reduction (PCA), and deep learning to develop a robust biomedical classification pipeline.

Dataset Description
Source: Microarray gene expression datasets (Affymetrix), accessible from the provided CSV files in the project directory or Google Drive.

Samples: Includes labeled samples from both training and independent (test) patient sets.

Example: actual.csv contains mappings of sample IDs to cancer labels (ALL/AML).

Features:

Thousands of gene expression measures per patient (e.g., 7,129 genes per sample).

Metadata columns: patient IDs and cancer type labels.

Environment & Dependencies
Programming Language: Python 3.6.4+

Key Libraries:

numpy, pandas – For data management and preprocessing.

scikit-learn – For splitting, scaling, and PCA.

tensorflow / keras – Model building and training.

matplotlib, seaborn – Visualization.

mpl_toolkits.mplot3d – 3D plotting (for PCA).

Execution Environment:

Compatible with Google Colab (utilizes google.colab.drive for cloud-based file access).

Pipeline & Methodology
Data Loading & Preparation

Mounts Google Drive and loads CSV feature and label files.

Splits patient label data into training and test sets aligned to gene expression samples.

Data are cleaned and reoriented for downstream analysis.

Preprocessing

Scaling: Applies StandardScaler for feature normalization.

Dimensionality Reduction: Uses PCA (Principal Component Analysis) to reduce feature space while retaining >95% variance for model input and interpretability.

Exploratory Visualization

Visualizes original and scaled data distributions.

Creates a 3D PCA projection of sample distribution colored by subtype.

Model Development

Builds a neural network classifier (Dense layers, ReLU activations, output for binary classification using Keras).

Early layers reflect post-PCA dimensions; network typically includes:

Dense(32, relu) → Dense(16, relu) → Output (sigmoid)

Sets random seeds for reproducibility.

Training & Validation

Model is trained for 200 epochs with stratified train/test splitting.

Assesses training/validation accuracy and loss at each epoch.

Evaluation

Tracks accuracy on held-out test samples.

Reports overall accuracy and insight into potential overfitting or data imbalance (observed moderate test accuracy and potential validation drop-off).

Key Results
The neural network demonstrates strong learning capability for training data, while test accuracy suggests classification remains non-trivial (likely due to high dimensionality, class imbalance, or limited sample size).

Distinct gene expression patterns for ALL and AML are visualized in PCA plots.

The workflow includes reproducible splits and a modular model, supporting extension or adaptation for further research.

Visualizations
Original vs. scaled feature distribution histograms.

3D PCA scatter plots color-coded by leukemia subtype.

Training/validation accuracy and loss curves over epochs.

Confusion matrix visualizations of final predictions.

How to Run
Clone the Repository

bash
git clone https://github.com/Adxrsh-17/Deep-Learning-Driven-Leukemia-Subtype-Classification-Using-Gene-Expression-Profiles.git
cd Deep-Learning-Driven-Leukemia-Subtype-Classification-Using-Gene-Expression-Profiles
Install Required Packages

bash
pip install numpy pandas scikit-learn tensorflow matplotlib seaborn
Prepare Data

Place the following files in the appropriate data directory or Google Drive path as specified in the notebook:

actual.csv

data_set_ALL_AML_train.csv

data_set_ALL_AML_independent.csv

Run the Notebook

Use Google Colab or Jupyter Notebook to open python_project.ipynb.

Execute each cell sequentially for a complete end-to-end workflow.

References
Microarray AML/ALL datasets and classical studies in gene-based cancer classification.

Open-source tools: TensorFlow, Scikit-learn, Pandas, Matplotlib.

For further background, see foundational literature on gene expression-based disease prediction and deep learning in bioinformatics.

This structured README.md provides clarity, professionalism, and aids in collaboration for both researchers and practitioners working on biomedical classification using gene expression data.
