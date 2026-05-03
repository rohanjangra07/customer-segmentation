# 🛍️ Mall Customer Segmentation using Hierarchical Clustering

## 📌 Project Overview

This project focuses on **customer segmentation** using **unsupervised machine learning** techniques. The objective is to group customers based on their purchasing behavior using:

* Age
* Income
* Spending Score

This helps businesses in:

* Targeted marketing 🎯
* Customer understanding 🧠
* Better decision making 📊

---

## 📂 Dataset Information

* Dataset: `Mall_Customers.csv`
* Total Records: 200

### Features:

| Feature       | Description            |
| ------------- | ---------------------- |
| CustomerID    | Unique ID of customer  |
| Gender        | Male/Female            |
| Age           | Customer age           |
| Income        | Annual Income (k$)     |
| SpendingScore | Spending score (1–100) |

---

## ⚙️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn
* SciPy

---

## 🔍 Project Workflow

### 1. Data Loading

```python
data = pd.read_csv("Mall_Customers.csv")
```

---

### 2. Data Understanding

* Checked shape → (200, 5)
* Used `describe()` for statistics
* Used `info()` for data types
* Checked null values → No missing data

---

### 3. Data Cleaning

Renamed columns for simplicity:

```python
data.rename(columns={
    'Annual Income (k$)': 'Income',
    'Spending Score (1-100)': 'SpendingScore'
}, inplace=True)
```

---

## 📊 Exploratory Data Analysis (EDA)

Performed multiple visualizations:

* 📈 Age Distribution
* 💰 Income Distribution
* 🛒 Spending Score Distribution
* 👥 Gender Count Plot
* 🔗 Pairplot (Feature Relationships)
* 🔥 Correlation Heatmap

### Key Insights:

* Spending score varies widely across customers
* Income and spending show clustering patterns
* Gender distribution is balanced

---

## 🔄 Data Preprocessing

Selected important features:

```python
x = data[['Age', 'Income', 'SpendingScore']]
```

### Standardization

* Mean = 0
* Standard Deviation = 1

```python
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
```

---

## 🌳 Hierarchical Clustering

### Dendrogram

Used **Ward linkage method** to determine cluster structure:

```python
dendrogram(linkage(x_scaled, method='ward'))
```

---

### Model Training

```python
model = AgglomerativeClustering(metric='euclidean', linkage='ward')
data['Cluster'] = model.fit_predict(x_scaled)
```

---

## 📈 Evaluation Metrics

### 🔹 Silhouette Score = 0.3179

* Measures how well clusters are separated
* Value range: -1 to 1
* Moderate clustering quality

### 🔹 Davies-Bouldin Index = 1.308

* Measures cluster compactness
* Lower value = Better clustering

---

## 📊 Results & Interpretation

* Customers are grouped into distinct clusters
* Each cluster represents a specific spending pattern
* Useful for:

  * High-value customer identification
  * Marketing strategy optimization
  * Business growth planning

---

## 🚀 How to Run

### Step 1: Clone Repository

```bash
git clone https://github.com/your-username/mall-customer-segmentation.git
```

### Step 2: Install Dependencies

```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy
```

### Step 3: Run Project

```bash
python main.py
```

---

## 📌 Future Enhancements

* Apply K-Means clustering
* Use Elbow Method for optimal clusters
* Add cluster visualization (scatter plots)
* Build a web app dashboard

---

## 👨‍💻 Author

**Rohan Jangra**

---

## ⭐ Support

If you found this project helpful, please give it a ⭐ on GitHub!
