import numpy as np
import pandas as pd
from sklearn.cluster import KMe

data = pd.read_csv(r"../../../rust/_garage/ccpp.csv")
X = np.array(data[data.columns[:]]) # can be removed if any


kmeans = KMeans(n_clusters=5)
y_kmeans = kmeans.fit_predict(X)
print(5,"- clusters")
print([i for i in y_kmeans])
print(kmeans.cluster_centers_)