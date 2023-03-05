# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn_extra.cluster import KMedoids
from scipy.spatial import Voronoi, voronoi_plot_2d

cmap = plt.cm.get_cmap('tab10')
# %% Load data
df = pd.read_csv('./worldcities.csv', sep=',')
country = 'Italy'
df_small = df[df.country == country][['city', 'lat', 'lng']]
X = df_small[['lng', 'lat']].to_numpy(dtype=float)

# %%
K = 5  # number of clusters
size = 10
fs = 15
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

ax[0].scatter(*X.T, color=cmap(0), s=size, alpha=0.8)
ax[0].set_title('Data', fontsize=fs)

kmeans = KMedoids(n_clusters=K, random_state=0)
kmeans.fit(X)
classes = kmeans.predict(X)
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
centroids = kmeans.cluster_centers_
vor = Voronoi(centroids)

# define color map
color_map = 255 * np.array(cmap.colors)[1:K+1]
data_3d = color_map[Z]
fig = voronoi_plot_2d(
    vor, ax=ax[1], show_points=False, show_vertices=False, line_colors='k',
    line_alpha=0.2)

ax[1].scatter(*X.T, color=cmap(classes + 1), s=size, alpha=0.15)
ax[1].scatter(
    centroids[:, 0],
    centroids[:, 1],
    marker="x",
    s=50,
    linewidths=3,
    color="k",
    zorder=10,
)

ax[1].imshow(
    data_3d,
    interpolation="nearest",
    extent=(xx.min(), xx.max(), yy.min(), yy.max()),
    aspect="auto",
    origin="lower",
    alpha=0.01
)

ax[1].set_title('K-medoids with K = {}'.format(K), fontsize=fs)
ax[1].set_xlim([x_min, x_max])
ax[1].set_ylim([y_min, y_max])


for i in range(centroids.shape[0]):
    lateqal = df_small['lat'] == centroids[i, 1]
    lngeqal = df_small['lng'] == centroids[i, 0]
    name = df_small[lngeqal & lateqal]['city'].values[0]
    ax[1].annotate(name, (centroids[i, 0], centroids[i, 1]+0.2),
                   fontsize=11, zorder=200)


plt.tight_layout()
plt.savefig('./kmedoids_{0}_{1}.pdf'.format(country, K))
# %%
