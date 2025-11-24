# 🌸 Iris Flower Classification – Machine Learning Project

This project demonstrates a complete Machine Learning pipeline using the classic Iris dataset from the UCI Machine Learning Repository.
It includes data loading, preprocessing, model training, evaluation, and prediction using Python & scikit-learn — all implemented inside a Jupyter Notebook.

📌 Project Overview

The Iris dataset is a small but widely used benchmark dataset containing:

150 samples

4 numerical features

Sepal Length

Sepal Width

Petal Length

Petal Width

3 classes

Iris-setosa

Iris-versicolor

Iris-virginica

The goal is to build and evaluate a machine learning model that can correctly classify the flower species based on the given features.

🧠 Machine Learning Workflow
✔ 1. Dataset Loading

The dataset is taken from the UCI ML Repository and loaded into a Pandas DataFrame.

✔ 2. Data Preprocessing

Check for missing values

Explore features

Encode target labels using LabelEncoder

Split into train/test sets

✔ 3. Model Training

You trained a classification model (RandomForest / DecisionTree / etc.). Example:

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(x.values, y.values)

✔ 4. Model Evaluation

Metrics used:

Accuracy Score

Confusion Matrix

Classification Report

✔ 5. Testing the Model

You tested the trained model using:

Manual inputs

Actual test dataset

New data samples

📊 Results

Your trained model achieved good accuracy on the Iris dataset (usually between 94–100%, depending on the algorithm).
All three Iris classes were correctly predicted in most test samples.


🚀 How to Run

Clone the repository:

git clone https://github.com/Krizhere/ML_Project1.git


Install dependencies:

pip install -r requirements.txt


Open the notebook:

jupyter notebook


Run all cells to train and test the model.

🛠 Technologies Used

Python

Jupyter Notebook

NumPy

Pandas

Scikit-learn

📚 Dataset Source

UCI Machine Learning Repository – Iris Dataset
(https://archive.ics.uci.edu/ml/datasets/iris
)

🙌 Acknowledgements

This project is created for learning and demonstrating basic machine learning workflows.
Feel free to fork, modify, or improve further!
