# %% Thanks to AdrienVannson https://github.com/AdrienVannson
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn_extra.cluster import KMedoids
from getcities import get_cities
cmap = plt.cm.get_cmap('tab10')
# %%
df = pd.read_csv('./worldcities.csv', sep=',')
country = 'France'
# keep only the n_max first cities (in term of number of inhabitants)
n_max = 1500
df_small = df[df.country == country][['city', 'lat', 'lng']].iloc[0:n_max]
X = df_small[['lng', 'lat']].to_numpy(dtype=float)
cities = [(name, lng, lat) for (name, lat, lng) in df_small.values]
cities_per_request = 100
D = get_cities(cities,
               cities_per_request=cities_per_request,
               verbose=True,
               force_recompute=True,  # force to recalculate all the distances, can be slow
               time_sleep=1e-3
               )
# %%
K = 4  # number of clusters
kmeans = KMedoids(n_clusters=K, random_state=0, metric='precomputed')
kmeans.fit(D)
classes = kmeans.labels_
centroids = X[kmeans.medoid_indices_]
# %%
size = 10
fs = 15
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].scatter(*X.T, color=cmap(0), s=size, alpha=0.8)
ax[0].set_title('Data', fontsize=fs)

ax[1].scatter(*X.T, color=cmap(classes + 1), s=size, alpha=0.3)
ax[1].scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="x",
    s=50,
    linewidths=3,
    color="k",
    zorder=10,
)
ax[1].set_title('K-medoids (shortest-path) with K = {}'.format(K), fontsize=fs)


for i in range(centroids.shape[0]):
    lateqal = df_small['lat'] == centroids[i, 1]
    lngeqal = df_small['lng'] == centroids[i, 0]
    name = df_small[lngeqal & lateqal]['city'].values[0]
    ax[1].annotate(name, (centroids[i, 0], centroids[i, 1]+0.2),
                   fontsize=11, zorder=200)

plt.tight_layout()
plt.savefig('./kmedoids_shortest_path_{0}_{1}.pdf'.format(country, K))
# %%
