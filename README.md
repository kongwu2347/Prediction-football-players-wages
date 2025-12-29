## ⚽ Football Player Wage Prediction with MLOps

[![Workflow Status](https://github.com/kongwu2347/Prediction-football-players-wages/actions/workflows/run_model.yml/badge.svg)](https://github.com/kongwu2347/Prediction-football-players-wages/actions/workflows/run_model.yml)

## 📖 Project Overview
This project predicts the wages of international football players using **XGBoost** and **automated hyperparameter tuning**. 

Unlike standard notebooks, this repository is engineered as a **production-ready pipeline** integrated with **GitHub Actions** for CI/CD.

## 🛠 Tech Stack & Engineering
* **Modeling:** XGBoost (Histogram-based method)
* **Optimization:** Optuna (Bayesian Optimization for hyperparameter tuning)
* **Preprocessing:** Scikit-Learn Pipelines & ColumnTransformers
* **CI/CD:** GitHub Actions (Automated training and prediction on push)
* **Data Handling:** Robust imputation strategies & Feature Engineering (Star Power, Growth Potential)

## 📊 Key Results
The model automatically optimizes for the best RMSE.
* **RMSE (Real Scale):** ~13,760 EUR (Typical performance)
* **R2 Score:** ~0.66+


