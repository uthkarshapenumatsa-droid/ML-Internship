import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

# ==========================================
# CUSTOMER SEGMENTATION USING K-MEANS
# ==========================================

print("="*50)
print("CUSTOMER SEGMENTATION USING K-MEANS")
print("="*50)

# Load Dataset
df = pd.read_csv("Mall_Customers.csv")

print("\nFirst 5 Rows")
print(df.head())

print("\nDataset Shape")
print(df.shape)

# Select Features
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

print("\nSelected Features")
print(X.head())

# ==========================================
# ELBOW METHOD
# ==========================================

wcss = []

for i in range(1,11):
    kmeans = KMeans(
        n_clusters=i,
        init='k-means++',
        random_state=42
    )

    kmeans.fit(X)

    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8,6))

plt.plot(range(1,11), wcss, marker='o')

plt.title("Elbow Method")

plt.xlabel("Number of Clusters")

plt.ylabel("WCSS")

plt.grid(True)

plt.savefig("elbow_method.png")

plt.show()

# ==========================================
# TRAIN MODEL
# ==========================================

kmeans = KMeans(
    n_clusters=5,
    init='k-means++',
    random_state=42
)

y_pred = kmeans.fit_predict(X)

df['Cluster'] = y_pred

print("\nCluster Count")

print(df['Cluster'].value_counts())

# ==========================================
# CUSTOMER CLUSTERS
# ==========================================

plt.figure(figsize=(10,8))

plt.scatter(
    X.iloc[:,0],
    X.iloc[:,1],
    c=y_pred,
    cmap='viridis',
    s=80
)

plt.scatter(
    kmeans.cluster_centers_[:,0],
    kmeans.cluster_centers_[:,1],
    color='red',
    marker='X',
    s=300,
    label='Centroids'
)

plt.title("Customer Segmentation")

plt.xlabel("Annual Income (k$)")

plt.ylabel("Spending Score (1-100)")

plt.legend()

plt.grid(True)

plt.savefig("customer_clusters.png")

plt.show()

print("\nCluster Centers")

print(kmeans.cluster_centers_)

print("\nTask 2 Completed Successfully!")
