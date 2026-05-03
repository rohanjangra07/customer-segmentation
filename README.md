# 🛍️ Mall Customer Segmentation using Clustering

## 📌 Project Overview
This project performs **customer segmentation** on mall customer data using **unsupervised machine learning techniques** like:
- Hierarchical Clustering
- Data Scaling
- Cluster Evaluation Metrics

The goal is to group customers based on:
- Age
- Income
- Spending Score

---

## 📂 Dataset
- Dataset used: `Mall_Customers.csv`
- Total records: 200
- Features:
  - CustomerID
  - Gender
  - Age
  - Annual Income (k$)
  - Spending Score (1–100)

---

## ⚙️ Technologies Used
- Python 🐍
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- SciPy

---

## 🔍 Steps Performed

### 1. Data Loading
```python
data = pd.read_csv("Mall_Customers.csv")
