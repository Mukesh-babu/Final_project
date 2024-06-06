
# Kannada MNIST Classification Project

This project is an extension of the classic MNIST classification problem. Instead of using Hindu numerals, we utilize a recently-released dataset of Kannada digits. This is a 10-class classification problem.

## Kannada Language

Kannada is a language spoken predominantly by people of Karnataka in southwestern India. The language has roughly 45 million native speakers and is written using the Kannada script. For more information, you can read the [Kannada Wikipedia page](https://en.wikipedia.org/wiki/Kannada).

## Dataset

The dataset can be downloaded from the following link: [Kannada-MNIST Dataset on Kaggle](https://www.kaggle.com/datasets/higgstachyon/kannada-mnist). All details of the dataset curation are captured in the paper titled: Prabhu, Vinay Uday. "Kannada-MNIST: A new handwritten digits dataset for the Kannada language." You can access the paper [here](https://arxiv.org/abs/1908.01242).

## Procedure

1. **Extract the Dataset**
   - Download the dataset from the provided Kaggle link.
   - Extract the dataset from the npz file. The dataset contains 60,000 images for training and 10,000 images for testing. Each image is of size 28x28.

2. **Principal Component Analysis (PCA)**
   - Perform PCA to reduce the dimensionality to 10 components. This transforms the train and test images into a 10-dimensional space instead of the original 28x28 dimension.

3. **Apply Classification Models**
   - Implement and train the following models on the transformed dataset:
     - Decision Trees
     - Random Forest
     - Naive Bayes Model
     - K-NN Classifier
     - SVM

4. **Evaluate Models**
   - For each model, compute the following metrics:
     - Precision
     - Recall
     - F1-Score
   - Generate and analyze the following visualizations:
     - Confusion Matrix
     - ROC-AUC Curve

5. **Experiment with Different PCA Components**
   - Repeat the above procedure with different PCA component sizes: 15, 20, 25, 30.

## Repository Structure

Kannada-MNIST-Classification/
├── data/
│ ├── train.npz
│ ├── test.npz
├── notebooks/
│ ├── data_preparation.ipynb
│ ├── pca_reduction.ipynb
│ ├── model_training.ipynb
│ ├── evaluation_metrics.ipynb
├── src/
│ ├── data_preparation.py
│ ├── pca_reduction.py
│ ├── model_training.py
│ ├── evaluation_metrics.py
├── README.md
├── requirements.txt
└── LICENSE

bash
Copy code

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Kannada-MNIST-Classification.git
   cd Kannada-MNIST-Classification
Install the required packages:
bash
Copy code
pip install -r requirements.txt
Usage
Prepare the Data

Run the data preparation script/notebook to extract and preprocess the data.
PCA Reduction

Execute the PCA reduction script/notebook to reduce the dataset to the desired number of components.
Train Models

Run the model training script/notebook to train the classifiers on the reduced dataset.
Evaluate Models

Execute the evaluation script/notebook to compute and visualize the performance metrics.
Contributing
Contributions are welcome! Please open an issue or submit a pull request for any features, enhancements, or bug fixes.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Feel free to customize the repository structure, installation instructions, and usage based on the actual implementation details of your project.

