# Credit Card Approval Prediction

This project builds a binary classification model to predict the approval status of credit card applications using machine learning techniques. It follows a structured deep learning workflow (DLWP 4.5 Universal Workflow) and evaluates performance using metrics like accuracy and AUC.


## Objectives

- Predict approval status of credit card applications using deep learning.
- Compare different model architectures to assess overfitting, generalisation, and performance.
- Apply dropout, early stopping, and tuning techniques in a structured experimental workflow.


## Dataset

- **Source**: [Kaggle – Credit Card Approval Dataset](https://www.kaggle.com/rikdifos/credit-card-approval-prediction)
- Contains categorical and numerical attributes like:
  - Age, Gender, Income, Credit Score, Marital Status, Education, etc.
- Target label: `approved` (Yes/No)


## Preprocessing Steps

- Converted categorical attributes using one-hot encoding
- Normalised numerical features using MinMaxScaler
- Split data into **80% training**, **20% test**
- Ensured class balance for fair evaluation


## Model Architectures

| Model                      | Type             | Description                                                                 |
|---------------------------|------------------|-----------------------------------------------------------------------------|
| **Baseline Model**        | DL (Shallow NN)  | 2 dense layers with ReLU, minimal capacity                                 |
| **Logistic Regression**   | Classical ML      | Interpretable linear classifier using TF-IDF-like inputs                   |
| **Random Forest**         | Classical ML      | Tree-based ensemble with 100 estimators                                    |
| **Deeper Neural Network** | DL                | 3–4 dense layers with more depth to capture complex interactions           |
| **Wider Neural Network**  | DL                | Fewer layers, but with more neurons per layer to increase model capacity   |


## Evaluation Tools

- Confusion Matrix
- AUC-ROC Curve
- Learning Curves (training vs validation loss)
- Model summary and layer inspection
