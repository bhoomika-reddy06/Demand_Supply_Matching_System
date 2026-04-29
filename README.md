# 🚚 Demand-Supply Matching System 

## 📌 Overview

This project builds an **AI-based Demand-Supply Matching System** for a logistics/delivery platform using the Uber Demand Dataset.
The system predicts demand across locations and time, and dynamically adjusts supply to improve efficiency.

---

## 🎯 Problem Statement

Delivery platforms often face:

* 🚫 High demand → shortages, delays, cancellations
* 📉 Low demand → idle resources, higher costs

This project solves that by:

* Predicting demand
* Optimizing supply allocation
* Reducing mismatches between demand and supply

---

## ⚙️ Technologies Used

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn

---

## 📊 Key Features

### 🔹 1. Demand Prediction

* Uses **Random Forest Regression**
* Predicts demand based on:

  * Time (hour, weekday)
  * Location (City / Airport)
  * Historical demand (lag features)

---

### 🔹 2. Feature Engineering

* Hour of day
* Weekday
* Lag features (previous demand)
* Rolling average

---

### 🔹 3. Model Evaluation

* Time-based cross-validation using **TimeSeriesSplit**
* Compared with baseline **Linear Regression**
* Performance measured using **RMSE**

---

### 🔹 4. Demand Clustering

* Uses **K-Means Clustering**
* Groups demand into:

  * High demand
  * Medium demand
  * Low demand

---

### 🔹 5. Dynamic Supply Optimization

* Supply is adjusted based on predicted demand and clusters:

  * High demand → increase supply
  * Medium demand → moderate supply
  * Low demand → reduce supply

---

### 🔹 6. Strategy Decision System

* Calculates demand-supply gap
* Recommends:

  * Increase Supply
  * Reduce Supply
  * Balanced

---

## 📈 Visualizations

* 📊 Demand trend by hour
* 🔥 Heatmap (location vs time)
* 📉 Actual vs Predicted demand
* 🧩 Cluster visualization
* 📦 Strategy distribution

---

## 📊 Results & Insights

* Peak demand occurs during **evening hours**
* Airport demand is higher in **morning**
* City demand is higher in **evening**
* Model captures demand trends with **moderate error (RMSE)**
* Most time periods are **balanced**, with some shortages
* Dynamic allocation improves efficiency and reduces mismatch

---

## 📂 Dataset

* Uber Request Dataset (Kaggle)

---

## 🎯 Project Outcomes

✔ Demand prediction
✔ Supply optimization
✔ RMSE evaluation
✔ Demand trend analysis
✔ Actionable business insights

---

## 🔮 Future Improvements

* Real-time demand prediction
* Weather/event-based features
* Advanced optimization algorithms
* Deployment as a web application

---

## 👩‍💻 Author

**Bhoomika Reddy M E**

---

## ⭐ Final Note

This project demonstrates how machine learning and data analysis can be used to build a **real-world demand-supply optimization system** that improves operational efficiency and customer experience.
