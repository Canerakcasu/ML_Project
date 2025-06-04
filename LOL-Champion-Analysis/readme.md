# 🧠 League of Legends Champion Analysis and Role Prediction

**📅 Project Date:** June 4, 2025  
**👤 Author:** Caner AKCASU

---

## 📝 Abstract

This project presents an end-to-end machine learning workflow to **predict a League of Legends champion's role** based on their in-game attributes. The pipeline includes:

- Data loading & exploration  
- Extensive exploratory data analysis (EDA)  
- Data cleaning and preprocessing  
- Model training with a **RandomForestClassifier**  
- Hyperparameter tuning  
- Model evaluation and visualization

All generated visualizations are automatically saved to a local directory.

---

## 🎯 Project Overview

The **goal** is to classify champions into their respective roles using their statistical attributes. The target variable is the `Role` column.

### 🔄 Workflow Summary:

- **Data Loading and Exploration**
  - Viewing head rows, info, missing values, and summary statistics
- **EDA (Exploratory Data Analysis)**
  - Histograms with KDE for numeric attributes (e.g., HP, Attack Damage)
  - Correlation matrix for numeric relationships
  - Count plots for categorical features (e.g., Role, Resource Type)
- **Data Cleaning and Preprocessing**
  - Impute missing values (numerical: median, categorical: mode)
  - Encode `Role` using LabelEncoder
  - Remove rare roles (less than 2 occurrences)
  - Apply StandardScaler and OneHotEncoder
- **Train-Test Split**
  - 80/20 split with stratification
- **Model Training and Optimization**
  - Train a RandomForestClassifier
  - Use GridSearchCV for tuning (`n_estimators`, `max_depth`, `min_samples_leaf`)
- **Evaluation**
  - Test set performance analysis
  - Confusion matrix visualization

---

## 📊 Dataset

- **File Name:** `champions.csv`  
- **Source:** [Kaggle - League of Legends Champions Dataset](https://www.kaggle.com/datasets/cutedango/league-of-legends-champions)

*Note: In the project script, the dataset is loaded from a local path.*

---

## 🔍 Key Findings (EDA Highlights)

- **Attack Range** showed a bimodal distribution separating melee vs. ranged champions.
- **Role distribution** is imbalanced – some roles dominate, others are rare combinations.
- **Mana** is the most frequently used resource type across champions.

---
