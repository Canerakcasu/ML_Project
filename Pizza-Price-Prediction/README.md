# 🍕 Pizza Price Prediction Project

## 📌 Project Overview

This project aims to **predict the price of pizzas** based on features such as size, restaurant, and extra toppings. Developed as a final project for a **Basics of Machine Learning** course, the objective is to explore various **regression algorithms**, evaluate their performance, and understand the **key factors influencing pizza prices** using the [Pizza-Price.csv](https://www.kaggle.com/datasets/alyeasin/predict-pizza-price) dataset from Kaggle.

---

## 📂 Dataset

- **Name:** `Pizza-Price.csv`
- **Source:** Kaggle  
  ➤ [Predicting Pizza Price for Beginners](https://www.kaggle.com/datasets/alyeasin/predict-pizza-price)
- **Description:** A small, beginner-friendly dataset for regression tasks.

### 🔑 Key Features
> *Column names may differ slightly; verify using `print(df.columns)`.*

- `Restaurant`: Name of the restaurant (Categorical)
- `Extra Cheeze`: Indicates extra cheese (Categorical: Yes/No)
- `Extra Mushroom`: Indicates extra mushrooms (Categorical: Yes/No)
- `Size by Inch`: Pizza diameter in inches (Numerical)
- `Extra Spicy`: Indicates whether the pizza is spicy (Categorical: Yes/No)
- `Price`: Target variable — pizza price (Numerical)

---

## 🧠 Methodology

### 1. Data Loading and Exploration
- Load dataset using **pandas**
- Inspect:
  - Column names
  - Data types
  - Missing values
  - Statistical summary
- Visualize distributions & relationships using **Matplotlib** and **Seaborn**

### 2. Data Preprocessing
- Identify:
  - Numerical features
  - Categorical features
- Apply:
  - `StandardScaler` to numerical features
  - `OneHotEncoder` to categorical features
- Use `ColumnTransformer` to apply transformations

### 3. Model Training and Hyperparameter Tuning
- Train/Test split
- Train multiple regression models:
  - Linear Regression
  - Decision Tree Regressor
  - Random Forest Regressor
  - Support Vector Regressor (SVR)
- Use `GridSearchCV` for hyperparameter tuning
- Wrap everything in **scikit-learn Pipelines**

### 4. Model Evaluation
- Evaluate using:
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Error (MAE)
  - R² Score
- Compare performance across all models

### 5. Results Visualization
- Plot predicted vs actual values
- Visualize residuals (errors)

---

## 🛠️ Technologies & Libraries

- Python 3.x
- **pandas** – Data handling
- **NumPy** – Numerical operations
- **scikit-learn** – ML models, preprocessing, pipelines
- **Matplotlib & Seaborn** – Visualization

---

📈 Example Results