import matplotlib.pyplot as plt
from MainData import MainData

def main():
    print("Введите кол-во кластеров для KMeans")
    n_clus1 = input()
    print("Введите кол-во кластеров для Agglomerative")
    n_clus2 = input()
    data = MainData()
    plt.figure(figsize=(15, 10))
    data.KMeans(plt, int(n_clus1))
    data.AffinityPropagation(plt)
    # data.MeanShift(plt)
    data.AgglomerativeClustering(plt, int(n_clus2))
    data.DBSCAN(plt)
    plt.show()

if __name__ == '__main__':
    main()
