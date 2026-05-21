import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

data = pd.read_csv('Mall_Customers.csv')

print(data.head())

X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

kmeans = KMeans(n_clusters=5, random_state=42)

kmeans.fit(X)

data['Cluster'] = kmeans.predict(X)

print(data.head())

plt.scatter(
    X['Annual Income (k$)'],
    X['Spending Score (1-100)'],
    c=data['Cluster']
)


plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=300,
    c='red'
)


plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.title('Mall Customer Segmentation')


plt.show()