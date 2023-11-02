import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the dataset
url = "https://raw.githubusercontent.com/renjithcg/liver_patient_dataset/main/indian_liver_patient.csv"
data = pd.read_csv(url)

# a. Analyze the dataset
print("a. Dataset Analysis:")
print(data.head())

# b. Statistical Summary
print("\nb. Statistical Summary:")
print(data.describe())

# c. Find the optimum number of clusters using the Elbow method
X = data.drop(columns=['Gender'])  # Remove non-numeric column
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow graph
plt.figure(figsize=(10,6))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Based on the Elbow method, choose an optimal number of clusters (let's say k=3 for example)
optimal_k = 3

# d. Perform subdivision of dataset into subgroups using K-means unsupervised technique
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=0)
data['Cluster'] = kmeans.fit_predict(X)

# e. Analyze the clusters
print("\ne. Analyze the Clusters:")
for i in range(optimal_k):
    cluster_data = data[data['Cluster'] == i]
    print(f"\nCluster {i+1} Statistics:")
    print(cluster_data.describe())

# Optional: You can save the clustered data to a CSV file
data.to_csv('clustered_liver_data.csv', index=False)
