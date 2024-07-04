# Customer Segmentation Using K-Means Clustering for Mall Customers

## Overview

This project aims to perform customer segmentation using the K-Means clustering algorithm on the Mall Customers dataset. The dataset contains information about customers, including their gender, age, annual income, and spending score. By clustering the customers, we can identify different groups of customers with similar characteristics, which can help in targeted marketing and improving customer service.

## Dataset

The dataset used in this project is available on Kaggle: [Mall Customers Dataset](https://www.kaggle.com/datasets/shwetabh123/mall-customers). 

## Requirements

Before you begin, ensure you have the following libraries installed:

```bash
numpy
pandas
matplotlib
seaborn
scikit-learn
```

You can install these libraries using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Steps

### 1. Import Libraries

First, we import the necessary libraries:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
```

### 2. Load Dataset

Download the dataset from Kaggle and load it into a DataFrame:

```python
df_customers = pd.read_csv('Mall_Customers.csv')
```

### 3. Data Exploration

- Display the number of records and features:

```python
print("Number of records:", df_customers.shape[0])
print("Number of features:", df_customers.shape[1])
```

- Display the feature names:

```python
print("Feature names:", df_customers.columns.tolist())
```

- Display the information about the DataFrame:

```python
df_customers.info()
```

- Display the numerical description of the DataFrame:

```python
df_customers.describe()
```

### 4. Data Preprocessing

- Check for missing values:

```python
print("Missing values:\n", df_customers.isnull().sum())
```

- Create dummy variables for 'Genre':

```python
gender = pd.get_dummies(df_customers['Genre'], drop_first=True)
df_customers = pd.concat([df_customers, gender], axis=1)
```

- Remove the features 'Genre' and 'CustomerID':

```python
df_customers.drop(['Genre', 'CustomerID'], axis=1, inplace=True)
```

### 5. Data Visualization

Explore the data using scatter plots:

```python
plt.subplots_adjust(right=2.0)

plt.subplot(1, 3, 1)
plt.scatter(df_customers["Age"], df_customers["Spending Score (1-100)"])
plt.xlabel("Age")
plt.ylabel("Spending Score (1-100)")

plt.subplot(1, 3, 2)
plt.scatter(df_customers["Age"], df_customers["Annual Income (k$)"])
plt.xlabel("Age")
plt.ylabel("Annual Income (k$)")

plt.subplot(1, 3, 3)
plt.scatter(df_customers["Annual Income (k$)"], df_customers["Spending Score (1-100)"])
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")

plt.show()
```

### 6. Model Building

- Finding the optimal number of clusters using the Elbow Method:

```python
wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, init='k-means++', random_state=42)
    km.fit(df_customers)
    wcss.append(km.inertia_)
    
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
```

- Fit the K-Means model:

```python
km = KMeans(n_clusters=5)
km.fit(df_customers)
df_customers["clusters"] = km.labels_

print('No. of data objects in each cluster')
print(df_customers['clusters'].value_counts())

print('Centroids of the clusters assigned')
print(km.cluster_centers_)
```

### 7. Visualizing the Clusters

Visualize the clusters:

```python
plt.scatter(df_customers["Spending Score (1-100)"], df_customers["Annual Income (k$)"], c=df_customers["clusters"])
plt.xlabel("Spending Score (1-100)")
plt.ylabel("Annual Income (k$)")
plt.show()
```

### 8. Evaluation of the Model

Evaluate the model using the Sum of Squared Error (SSE):

```python
sse = km.inertia_
print('Sum of Squared Error (sse) =', sse)
```

## Conclusion

This project demonstrates how to perform customer segmentation using K-Means clustering. By analyzing the clusters, businesses can better understand their customers and tailor their marketing strategies accordingly.


