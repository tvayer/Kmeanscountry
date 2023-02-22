#%%
import matplotlib.pyplot as plt
cmap = plt.cm.get_cmap('tab10')
import numpy as np
from scipy.spatial.distance import cdist
from sklearn_extra.cluster import KMedoids
from scipy.spatial import Voronoi, voronoi_plot_2d
import pandas as pd
#%% Load data
df = pd.read_csv('./worldcities.csv', sep=',')
country = 'France'
df_small = df[df.country == country][['city','lat','lng']]
X = df_small[['lng','lat']].to_numpy(dtype=float)

#%%
K = 5 #number of clusters
size = 10
fs = 15
fig, ax = plt.subplots(1,2, figsize = (10,5))
cmap = plt.cm.get_cmap('tab10')

ax[0].scatter(X[:,0], X[:,1], color= cmap(0), s=size, alpha=0.8)
ax[0].set_title('Data', fontsize = fs)

kmeans = KMedoids(n_clusters=K, random_state=0)
kmeans.fit(X)
classes = kmeans.predict(X)
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])
centroids = kmeans.cluster_centers_
vor = Voronoi(centroids)
Z = Z.reshape(xx.shape)
# define color map 
color_map = {i: 255*np.array(cmap.colors[i+1]) for i in range(K)}
data_3d = np.ndarray(shape=(Z.shape[0], Z.shape[1], 3), dtype=int)
for i in range(0, Z.shape[0]):
    for j in range(0, Z.shape[1]):
        data_3d[i][j] = color_map[Z[i][j]]
fig = voronoi_plot_2d(vor, ax=ax[1], show_points=False,show_vertices=False, line_colors='k', 
                        line_alpha =0.2)

ax[1].scatter(X[:,0], X[:,1], color= [cmap(classes[i]+1) for i in range(X.shape[0])], s=size, alpha=0.3)
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
    alpha = 0.01
)

ax[1].set_title('K-medois with K = {}'.format(K), fontsize = fs)
ax[1].set_xlim([x_min,x_max])
ax[1].set_ylim([y_min,y_max])


for i in range(centroids.shape[0]):
    lateqal = df_small['lat'] == centroids[i,1] 
    lngeqal = df_small['lng'] == centroids[i,0] 
    name = df_small[lngeqal & lateqal]['city'].values[0]
    ax[1].annotate(name, (centroids[i,0], centroids[i,1]+0.2), fontsize = 11, zorder= 200)


plt.tight_layout()
plt.savefig('./kmedois_{0}_{1}.pdf'.format(country,K))
# %%
