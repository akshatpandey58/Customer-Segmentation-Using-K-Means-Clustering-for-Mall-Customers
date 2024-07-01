#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import requests


# In[2]:


df_customers = pd.read_csv(r'C:\Users\Acer\OneDrive\Documents\Mall_Customers.csv')


# In[3]:


print("Number of records and features:", df_customers.shape)


# In[4]:


print("Feature names:", df_customers.columns.tolist())


# In[5]:


print("Information about the dataframe:")
print(df_customers.info())


# In[6]:


print("Numerical description of the dataframe:")
print(df_customers.describe())


# In[7]:


print("Missing values in the dataset:")
print(df_customers.isnull().sum())


# In[8]:


gender = pd.get_dummies(df_customers['Genre'], drop_first=True)


# In[9]:


df_customers = pd.concat([df_customers, gender], axis=1)


# In[10]:


df_customers.drop(['Genre', 'CustomerID'], axis=1, inplace=True)


# In[11]:


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


# In[12]:


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


# In[13]:


km = KMeans(n_clusters=5, random_state=42)
km.fit(df_customers)
df_customers["clusters"] = km.labels_

print('No. of data objects in each cluster:')
print(df_customers['clusters'].value_counts())

print('Centroids of the clusters assigned:')
print(km.cluster_centers_)


# In[14]:


plt.scatter(df_customers["Spending Score (1-100)"], df_customers["Annual Income (k$)"], c=df_customers["clusters"])
plt.xlabel("Spending Score (1-100)")
plt.ylabel("Annual Income (k$)")
plt.show()


# In[15]:


sse = km.inertia_
print('Sum of Squared Error (SSE) =', sse)


# In[ ]:




