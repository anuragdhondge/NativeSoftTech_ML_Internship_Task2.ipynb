# Titanic Survival Prediction with Random Forest

This repository contains a complete workflow for predicting Titanic survival using the Random Forest classifier and Scikit-learn. The project includes data preprocessing, model training, hyperparameter tuning, and evaluation.

## Table of Contents

- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Workflow](#workflow)
  - [1. Data Preprocessing](#1-data-preprocessing)
  - [2. Model Training](#2-model-training)
  - [3. Hyperparameter Tuning](#3-hyperparameter-tuning)
  - [4. Model Evaluation](#4-model-evaluation)
- [Results](#results)
- [How to Use](#how-to-use)

## Dataset

The dataset used in this project is the Titanic dataset. It includes features such as:
- Passenger demographics (e.g., age, gender)
- Ticket class
- Number of siblings/spouses aboard
- Number of parents/children aboard
- Embarked location, and more.

## Project Structure

```
|-- titanic_rf_analysis
    |-- data
        |-- Titanic Dataset For Native SoftTech.csv
    |-- src
        |-- preprocess.py
        |-- train_model.py
        |-- tune_model.py
        |-- evaluate_model.py
    |-- README.md
```

## Installation

1.Set up your Python environment using Anaconda Navigator:

Open Anaconda Navigator.
Create a new environment (e.g., titanic_env) with Python 3.8 or higher.
Activate the environment:
bash
(conda activate titanic_env)
Install required libraries in the environment:
bash
(conda install pandas numpy scikit-learn)
Alternatively, install the required files and libraries manually:

Ensure you have Python installed (3.8 or higher).
Install the required libraries using pip:
bash
(pip install pandas numpy scikit-learn)
If you prefer using Jupyter Notebook:

Install Jupyter Notebook in your environment:
bash
(conda install notebook)
Launch Jupyter Notebook:
bash
(jupyter notebook)
This updated section is tailored for users who prefer Anaconda Navigator or Jupyter Notebook for managing their Python environments.
   
## Workflow

### 1. Data Preprocessing

- Fill missing values for `Age`, `Embarked`, and `Fare`.
- Encode categorical variables `Sex` and `Embarked` using `LabelEncoder`.
- Drop unnecessary columns: `Name`, `Ticket`, and `Cabin`.
- Split the data into training and testing sets using `train_test_split`.

### 2. Model Training

- Initialize a `RandomForestClassifier` with a random seed.
- Train the model on the training dataset.

### 3. Hyperparameter Tuning

- Use `GridSearchCV` to perform hyperparameter tuning for the Random Forest model.
- Hyperparameter grid:
  ```python
  param_grid = {
      'n_estimators': [50, 100, 150],
      'max_depth': [None, 10, 20, 30],
      'min_samples_split': [2, 5, 10],
      'min_samples_leaf': [1, 2, 4]
  }
  ```

- Perform 5-fold cross-validation to find the best parameters.

### 4. Model Evaluation

- Evaluate the best model on the test dataset using:
  - Accuracy
  - Classification report
  - Confusion matrix

## Results

### Best Parameters:
```
{'max_depth': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 50}
```

### Test Accuracy:
```
Accuracy on Test Set: 1.0
```

### Classification Report:
```
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        50
           1       1.00      1.00      1.00        34

    accuracy                           1.00        84
   macro avg       1.00      1.00      1.00        84
weighted avg       1.00      1.00      1.00        84
```

### Confusion Matrix:
```
[[50  0]
 [ 0 34]]
```

## How to Use

1. Place the Titanic dataset in the `data` directory.
2. Run the preprocessing script:
   ```bash
   python src/preprocess.py
   ```
3. Train the model:
   ```bash
   python src/train_model.py
   ```
4. Tune the hyperparameters:
   ```bash
   python src/tune_model.py
   ```
5. Evaluate the model:
   ```bash
   python src/evaluate_model.py
   
