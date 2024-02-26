# credit-risk-classification

This repository contains code for credit risk classification using machine learning techniques in Python. The analysis is based on the lending data provided in the lending_data.csv file and implemented in the credit_risk_classification.ipynb Jupyter Notebook.

## Overview
The credit risk classification project involves predicting whether a loan is high-risk or healthy based on various features such as loan size, interest rate, borrower income, debt-to-income ratio, number of accounts, derogatory marks, and total debt. The goal is to develop a machine learning model that accurately classifies loans to assist in risk assessment and decision-making processes.

## Data Preparation
The lending data is loaded from the lending_data.csv file into a Pandas DataFrame. Before training the machine learning model, the data is preprocessed and split into training and testing sets using train_test_split from scikit-learn. Additionally, the features are standardized using StandardScaler to ensure uniform scaling.

## Model Training and Evaluation
A logistic regression model is employed for credit risk classification due to its interpretability and effectiveness in binary classification tasks. The model is trained on the training dataset and evaluated using the testing dataset. Evaluation metrics such as accuracy score and confusion matrix are computed to assess the model's performance.

## Analysis
The logistic regression model achieves a high accuracy score of approximately 99.18% on the testing dataset. Further analysis of the confusion matrix reveals 563 true positives (TP), 18663 true negatives (TN), 102 false positives (FP), and 56 false negatives (FN). Based on this data, the model will predict correctly a high-risk loan 84.66% of the time against 99.07% for healthy loan. it is important to be mindfull that the number of healthy loans feed to the model are 33x higher than for high risk loans.

## Files
Under Credit_Risk Folder:
credit_risk_classification.ipynb: Jupyter Notebook containing the code for credit risk classification.
Under Resources: lending_data.csv: Input data file containing lending information.
README.md: Overview of the project.

## Usage
To replicate the analysis:

Ensure you have Python and Jupyter Notebook installed on your system.
Clone this repository to your local machine.
Open the credit_risk_classification.ipynb notebook in Jupyter Notebook.
Execute the cells in the notebook to load the data, preprocess it, train the model, and evaluate its performance.
Review the results and metrics obtained from the analysis.

## Dependencies
The following Python libraries are required to run the code:

pandas
scikit-learn