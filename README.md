# Telco Customer Churn Classification

![](https://github.com/sujikathir/Telco-Customer-Churn-Prediction/blob/main/Images/Cover%20pic.jpg)

## Table of Contents
- [Introduction](#introduction)
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Feature Scaling](#feature-scaling)
- [Model Development](#model-development)
- [Results and Evaluation](#results-and-evaluation)
- [Conclusion](#conclusion)
- [Key Concepts](#key-concepts)
- [Project Structure](#project-structure)

## Introduction
This project focuses on predicting customer churn using machine learning techniques. A crucial aspect of this project is the data preprocessing stage, with a particular emphasis on feature scaling and its impact on model performance.

## Project Overview
The goal is to develop a model that can predict whether a customer is likely to churn based on various attributes. The project demonstrates the importance of proper data preprocessing, including handling missing values, encoding categorical variables, and scaling features.

## Project Structure

**Data Exploration**
- Dataset characteristics
- Visualization of feature distributions

**Data Preprocessing**
- Handling missing values
- Dealing with categorical variables

**Feature Scaling Implementation**
- MinMaxScaler
- StandardScaler
- RobustScaler

**Visualization of Scaling Effects**

**Model Training and Evaluation**
- Comparison of model performance with different scaling techniques

**Analysis of Results**

**Best Practices and Recommendations**

## Dataset
The dataset includes information about:
- **Customer demographics** (gender, age range, partners, dependents)
- **Services each customer has signed up for** (phone, internet, online security, etc.)
- **Customer account information** (tenure, contract type, payment method, etc.)
- **Billing information** (monthly charges, total charges)

Key characteristics:
- No missing values in most columns
- 'TotalCharges' column contains some empty strings
- 'CustomerID' column is not needed for prediction

## Data Preprocessing
### Handling Missing Values:
- Identified and addressed empty strings in 'TotalCharges'
- Decision made to drop rows with missing values (11 rows) due to small number

### Feature Engineering:
- Removed 'CustomerID' column
- Encoded categorical variables

### Dealing with Duplicates:
- Checked and removed any duplicate entries

## Feature Scaling
This project explores different scaling techniques and their impact on the churn prediction model:

### StandardScaler (Standardization):
- Used for features with unknown distribution
- Helps in handling outliers better than normalization

### MinMaxScaler (Normalization):
- Applied to bring features to a [0,1] range
- Useful when the boundaries of features are known

### RobustScaler:
- Utilized for its robustness to outliers
- Especially useful for 'TotalCharges' which showed high skewness

Key considerations:
- Scaling applied after train-test split to prevent data leakage
- Comparative analysis of model performance with different scalers

## Model Development
- Feature selection based on correlation analysis
- Train-test split of the dataset
- Model selection (e.g., Logistic Regression, Random Forest, etc.)
- Model training with different scaled datasets
- Hyperparameter tuning

## Results and Evaluation
- Comparison of model performance with different scaling techniques
- Analysis of feature importance
- Evaluation metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC

## Conclusion
This project demonstrates the critical role of data preprocessing, particularly feature scaling, in customer churn prediction. Key findings include:
- The impact of different scaling techniques on model performance
- The importance of handling skewed features like 'TotalCharges'
- Best practices for preprocessing in churn prediction tasks

The insights gained from this project can be applied to improve customer retention strategies and optimize business operations.

## Key Concepts
### Standardization vs. Normalization
- **Standardization (Z-score normalization):** Rescales features to have a mean of 0 and a standard deviation of 1. It's not bounded by a specific range.
- **Normalization (MinMaxScaler):** Rescales features to a fixed range, typically [0,1] or [-1,1].

### Impact on Machine Learning Algorithms
- **Gradient Descent Algorithms:** Linear regression, logistic regression, and neural networks benefit from scaled features as it helps in faster convergence.
- **Distance-Based Algorithms:** K-Nearest Neighbors (KNN), K-means clustering, and Support Vector Machines (SVM) are highly sensitive to feature scales.
- **Tree-Based Algorithms:** Generally robust to feature scaling, but can still benefit in some cases.

### Scaling Techniques
- **MinMaxScaler (Normalization):**
  ```python
  X_scaled = (X - X_min) / (X_max - X_min)


## Use cases:

- When the upper and lower boundaries of features are known
- In image processing, where pixel intensities need to be normalized

### StandardScaler (Standardization):

 ```python

X_scaled = (X - μ) / σ
```

Where μ is the mean and σ is the standard deviation.

**Use cases:**

- When the distribution of features is unknown or varies significantly
- For algorithms assuming normally distributed data

### RobustScaler:

 ``` python

X_scaled = (X - median) / IQR
```

Where IQR is the Interquartile Range.

**Use cases:**

- When dealing with datasets containing significant outliers
- For preserving the relative relationships between outliers and other data points

### Best Practices
- Scale after train-test split to prevent data leakage
- Consider scaling the target variable in regression problems
- Handle outliers carefully based on domain knowledge
- Choose the appropriate scaler for your data and algorithm
- Standardize non-normal distributions when appropriate
- Compare model performance with different scaling techniques


### Handling Outliers
The project explores the impact of outliers on different scaling techniques:

- MinMaxScaler's high sensitivity to outliers
- StandardScaler's moderate sensitivity
- RobustScaler's effectiveness in handling outliers

Strategies for dealing with outliers are discussed and demonstrated.

### Scaling and ML Algorithms

The project analyzes how different ML algorithms are affected by feature scaling:

- Gradient Descent based algorithms
- Distance-based algorithms
- Tree-based algorithms

Practical examples and performance comparisons are provided.



