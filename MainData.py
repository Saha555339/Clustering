from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN, AffinityPropagation, MeanShift
import numpy as np
import requests
from bs4 import BeautifulSoup as BS

URL = ''
Type_parser = 'html.parser'

class MainData:

    def __init__(self):
        # request = requests.get(URL)
        # html = BS(request.content, Type_parser)
        # my_list = []
        # for line in html.text.split("\n"):
        #     if not line.strip():
        #         continue
        #     my_list.append(line.lstrip().split())
        my_list = []
        for i in range(0,1000):
            x = np.random.randint(0,10000000)
            y = np.random.randint(0,10000000)
            my_list.append((x,y))
        arr = np.array(my_list)
        self.main_array = arr.astype(np.int32)

    def AgglomerativeClustering(self, plt, n_clus):
        plt.subplot(234)
        cluster = AgglomerativeClustering(n_clus, affinity='euclidean', linkage='ward')
        cluster.fit_predict(self.main_array)
        plt.scatter(self.main_array[:, 0], self.main_array[:, 1], c=cluster.labels_, cmap='rainbow')
        plt.title('Agglomerative Clustering')

    def AffinityPropagation(self, plt):
        plt.subplot(232)
        affinity = AffinityPropagation(preference=-50)
        affinity = affinity.fit(self.main_array)
        plt.scatter(self.main_array[:, 0], self.main_array[:, 1], alpha=0.7, edgecolors='b')
        plt.scatter(self.main_array[:, 0], self.main_array[:, 1], c=affinity.labels_, cmap='rainbow', alpha=0.7, edgecolors='b')
        plt.title('Affinity Propagation')

    def KMeans(self, plt, n_clus):
        plt.subplot(231)
        kmean = KMeans(n_clus).fit(self.main_array)
        y_kmeans = kmean.predict(self.main_array)
        plt.scatter(self.main_array[:, 0], self.main_array[:, 1], c=y_kmeans, s=10, cmap='rainbow')
        centers = kmean.cluster_centers_
        plt.scatter(centers[:, 0], centers[:, 1], c='black', s=50, alpha=1)
        plt.title('KMeans')

    def DBSCAN(self, plt):
        plt.subplot(235)
        Dbscan = DBSCAN(eps=20000, min_samples=5).fit(self.main_array)
        plt.scatter(self.main_array[:, 0], self.main_array[:, 1], c=Dbscan.labels_, cmap='plasma')
        plt.title('DBSCAN')

    def MeanShift(self, plt):
        plt.subplot(233)
        ms = MeanShift(max_iter=5, bandwidth=50000)
        ms.fit(self.main_array)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        colors = 10 * ['r.', 'g.', 'b.', 'c.', 'k.', 'y.', 'm.']
        for i in range(len(self.main_array)):
            plt.plot(self.main_array[i][0], self.main_array[i][1], colors[labels[i]], markersize=3)
        plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], marker=".", color='k', s=20, linewidths=5, zorder=10)
        plt.title('Mean Shift')
