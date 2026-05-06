import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

csv_path = "seeds_dataset.txt"
n_clusters = 3

df = pd.read_csv(csv_path, sep=r"\s+|,", engine="python")

if df.shape[1] == 1:
    df = pd.read_csv(csv_path, sep=r"\s+|,", engine="python", header=None)

if df.shape[1] == 8:
    df.columns = [
        "area","perimeter","compactness",
        "length_kernel","width_kernel",
        "asymmetry","groove_length","class"
    ]

df = df.apply(pd.to_numeric, errors="coerce")

df = df.fillna(df.median())

X = df.drop(columns=["class"], errors="ignore")

print("=== DATASET INFO ===")
print("Samples:", X.shape[0])
print("Features:", X.shape[1])
print("\nDescribe:\n", X.describe())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
labels = model.fit_predict(X_scaled)

sil_score = silhouette_score(X_scaled, labels)
inertia = model.inertia_

print("\n=== MODEL METRICS ===")
print("Inertia:", inertia)
print("Silhouette Score:", sil_score)

unique, counts = np.unique(labels, return_counts=True)
print("\nCluster sizes:")
for u, c in zip(unique, counts):
    print(f"Cluster {u}: {c}")

n_components = min(2, X_scaled.shape[0], X_scaled.shape[1])
pca = PCA(n_components=n_components)

X_pca = pca.fit_transform(X_scaled)

plt.figure()
if n_components == 2:
    centers_pca = pca.transform(model.cluster_centers_)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
    plt.scatter(centers_pca[:, 0], centers_pca[:, 1], marker='x')
    plt.xlabel("PC1")
    plt.ylabel("PC2")
else:
    plt.scatter(range(len(X_pca)), X_pca[:, 0], c=labels)
    plt.xlabel("Samples")
    plt.ylabel("PC1")

plt.title("KMeans Clustering (PCA projection)")
plt.show()

sample_silhouette_values = silhouette_samples(X_scaled, labels)

plt.figure()
y_lower = 10
for i in range(n_clusters):
    ith_cluster_values = sample_silhouette_values[labels == i]
    ith_cluster_values.sort()
    
    size_cluster = ith_cluster_values.shape[0]
    y_upper = y_lower + size_cluster
    
    plt.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_values)
    y_lower = y_upper + 10

plt.axvline(x=sil_score)
plt.title("Silhouette Plot")
plt.xlabel("Silhouette coefficient values")
plt.ylabel("Cluster label")
plt.show()

inertias = []
k_range = range(1, 10)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.figure()
plt.plot(k_range, inertias, marker='o')
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

df_clusters = pd.DataFrame(X_scaled, columns=X.columns)
df_clusters["cluster"] = labels

print("\n=== CLUSTER FEATURE MEANS ===")
print(df_clusters.groupby("cluster").mean())