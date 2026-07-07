# 🚚 Supply Chain Late Delivery Risk Prediction using Machine Learning

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11+-blue?style=for-the-badge\&logo=python)
![Pandas](<https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge\&logo=pandas>)
![Scikit-Learn](<https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=for-the-badge\&logo=scikitlearn>)
![Streamlit](<https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?style=for-the-badge\&logo=streamlit>)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</p>

---

<h2><a href="https://supplychain-prediction.streamlit.app/" target="_blank"> Application Link </a></h2>

---

# 📌 Project Overview

Late deliveries are one of the biggest challenges in supply chain management. Delayed shipments increase operational costs, reduce customer satisfaction, and negatively impact business performance.

This project uses **Machine Learning** to predict whether an order is likely to be delivered late based on historical supply chain data.

The project covers the complete Data Science lifecycle:

* Data Cleaning
* Exploratory Data Analysis (EDA)
* Feature Engineering
* Machine Learning
* Model Evaluation
* Hyperparameter Tuning
* Model Deployment with Streamlit

---

# 🎯 Problem Statement

The objective of this project is to build a classification model that predicts the **Late Delivery Risk** of an order before shipment.

Target Variable:

```
Late_delivery_risk
```

Prediction Classes:

* 0 → On-Time Delivery
* 1 → Late Delivery

---

# 📊 Dataset Information

The dataset contains supply chain information including:

* Customer Information
* Product Information
* Order Details
* Shipping Details
* Department Information
* Region Information
* Delivery Status
* Shipping Mode
* Sales & Profit
* Delivery Risk

---

# ⚙️ Project Workflow

```
Dataset
    │
    ▼
Data Cleaning
    │
    ▼
Feature Engineering
    │
    ▼
Exploratory Data Analysis
    │
    ▼
Train-Test Split
    │
    ▼
Machine Learning Models
    │
    ▼
Hyperparameter Tuning
    │
    ▼
Best Model Selection
    │
    ▼
Model Evaluation
    │
    ▼
Model Saving
    │
    ▼
Streamlit Deployment
```

---

# 📈 Exploratory Data Analysis

The notebook includes detailed visualizations such as:

* Missing Value Analysis
* Correlation Heatmap
* Class Distribution
* Shipping Mode Analysis
* Customer Segment Analysis
* Order Region Analysis
* Department Analysis
* Product Category Analysis
* Delivery Delay Distribution
* Profit Analysis
* Monthly Order Trend
* Hourly Order Trend
* Feature Correlation

---

# 🧠 Feature Engineering

The following preprocessing techniques were applied:

* Missing Value Treatment
* Duplicate Removal
* Date Feature Extraction
* Frequency Encoding
* Label Encoding
* One-Hot Encoding
* Feature Scaling (if required)
* Class Imbalance Handling using SMOTE

---

# 🤖 Machine Learning Models

The following algorithms were evaluated:

* Logistic Regression
* Decision Tree
* Random Forest
* Extra Trees Classifier
* XGBoost
* CatBoost
* LightGBM

---

# 📊 Model Evaluation Metrics

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC Score
* Confusion Matrix
* Classification Report
* Cross Validation

---

# 🏆 Best Model

After comparing multiple algorithms, the best-performing model was selected based on:

* Highest Accuracy
* Better Generalization
* ROC-AUC Score
* Cross Validation Performance

The final trained model is stored as:

```
model
```

---

# 🌐 Streamlit Web Application

The project includes a professional Streamlit application with:

* Modern Dashboard
* Prediction Interface
* Sidebar Navigation
* Model Information
* Probability Prediction
* Feature Description
* Responsive Design

Run the application:

```bash
streamlit run app.py
```

---

# 📂 Project Structure

```
SupplyChain-Late-Delivery-Prediction
│
├── data
│   └── SupplyChainDataset.csv
│
├── EDA
│	└── Supply_Chain_EDA.ipynb
│
├── model
│   ├── freq_mappings.pkl
│   └── rf_late_delivery_model.pkl
│
├── app.py
│
├── README.md
│
├── requirements.txt
```

---

# 🚀 Installation

## Clone Repository

```bash
git clone https://github.com/zainulabedeen589/SupplyChain-Late-Delivery-Prediction.git
```

---

## Create Virtual Environment

### Using uv

```bash
uv venv
```

Activate Environment

### Windows

```bash
.venv\Scripts\activate
```

### Linux / macOS

```bash
source .venv/bin/activate
```

---

## Install Dependencies

```bash
uv pip install -r requirements.txt
```

or

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Notebook

```bash
├── EDA
	└── Supply_Chain_EDA.ipynb
```

---

# ▶️ Run Streamlit App

```bash
streamlit run app.py
```

---

# 📦 Requirements

Main libraries used:

* pandas
* numpy
* matplotlib
* plotly
* seaborn
* scikit-learn
* xgboost
* lightgbm
* catboost
* imbalanced-learn
* streamlit
* joblib

---

# 📸 Screenshots

Add screenshots of:

* Dashboard
* Prediction Page
* EDA Charts
* Model Performance
* Streamlit UI

inside the **screenshots/** folder.

---

# 🔮 Future Improvements

* Real-time Prediction API
* Docker Support
* Cloud Deployment
* SHAP Explainability
* Model Monitoring
* Automated Retraining
* CI/CD Pipeline

---

# 👨‍💻 Author

**ZAINUL ABEDEEN**

AI Engineer | Machine Learning | Deep Learning | Data Science

Email: [zainulpasha589@gmail.com](mailto:zainulpasha589@gmail.com)

GitHub: [https://github.com/zainulabedeen589](https://github.com/zainulabedeen589)

LinkedIn: [https://linkedin.com/in/zainulabedeen589](https://linkedin.com/in/zainulabedeen589)

---

# ⭐ Support

If you found this project helpful, consider giving it a ⭐ on GitHub.

It helps the project reach more learners and motivates future improvements.

---
