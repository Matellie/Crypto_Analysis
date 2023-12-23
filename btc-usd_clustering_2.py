from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

import matplotlib.pyplot as plt
import numpy as np

def plot_BTC_USD(data):
    # Compute average price per day for plotting
    avg_price = [open + close / 2 for open, close in zip(data[:,0], data[:,3])]

    #Plot BTC-USD
    figure, axis = plt.subplots(2)
    figure.tight_layout(pad=3.0)

    axis[0].plot(avg_price)
    axis[0]
    axis[0].set_title("BTC-USD linear scale")

    axis[1].plot(avg_price)
    axis[1].set_yscale('log')
    axis[1].set_title("BTC-USD log scale")

    plt.show()

def plot_clusters_kmeans(data):
    # Compute clusters
    kmeans_2 = KMeans(n_clusters=2, random_state=0, n_init=10).fit(data)
    kmeans_5 = KMeans(n_clusters=5, random_state=0, n_init=10).fit(data)

    #Plot clusters
    figure, axis = plt.subplots(2, 1) # 2x1 grid
    figure.tight_layout(pad=3.0) # Space between plots

    axis[0].scatter(
        [a for a in range(len(data))], 
        [open + close / 2 for open, close in zip(data[:,0], data[:,3])], 
        c=kmeans_2.labels_,
        cmap='rainbow', s=1
        )
    axis[0].set_title("KMeans 2 clusters")

    axis[1].scatter(
        [a for a in range(len(data))], 
        [open + close / 2 for open, close in zip(data[:,0], data[:,3])], 
        c=kmeans_5.labels_,
        cmap='rainbow', s=1
        )
    axis[1].set_title("KMeans 5 clusters")

    plt.show()

def plot_PCA(data):
    # Compute PCA
    pca = PCA().fit(data)

    # Plot PCA explained variance ratio
    plt.bar(
        range(1, len(pca.explained_variance_ratio_)+1),
        pca.explained_variance_ratio_
        )
    plt.title("PCA explained variance ratio")
    plt.show()

def plot_dimension_impact(data):
    # Compute PCA
    pca = PCA().fit(data)
    print(pca.components_)

    # Plot PCA components
    plt.bar(
        range(1, len(pca.components_)+1),
        pca.components_[0]
        )
    plt.title("PCA dimension impact")
    plt.show()

def plot_clusters_price_volume(data):
    # Compute clusters
    kmeans_5 = KMeans(n_clusters=5, random_state=0, n_init=10).fit(data)

    # Plot
    figure, axis = plt.subplots(2, 1)
    figure.tight_layout(pad=3.0)

    # Plot clusters with price as y axis
    axis[0].scatter(
        [a for a in range(len(data))], 
        [open + close / 2 for open, close in zip(data[:,0], data[:,3])], 
        c=kmeans_5.labels_,
        cmap='rainbow', s=1.5
        )
    axis[0].set_title("y-axis: price")

    # Plot clusters with volume as y axis
    axis[1].scatter(
        [a for a in range(len(data))], 
        data[:,4], 
        c=kmeans_5.labels_,
        cmap='rainbow', s=1.5
        )
    axis[1].set_title("y-axis: volume")

    plt.show()

def main():
    # Load data
    """
    The data is from Yahoo Finance and is the daily price of BTC-USD.
    The columns are: Date, Open, High, Low, Close, Adj Close, Volume.
    We don't keep the date column and the adjusted close column.
    Hence, we have 5 dimensions: Open, High, Low, Close, Volume.
    """
    X = np.loadtxt("BTC-USD_daily.csv", delimiter=",", skiprows=1, usecols=(1,2,3,4,6))
    X = normalize(X, axis=0) # Normalize data

    # Plot BTC-USD in linear and log scale
    plot_BTC_USD(X)

    # Plot KMeans with 2 and 5 clusters
    plot_clusters_kmeans(X)
    """
    This time, the clusters seem to be separated by price range.
    However, we can note that they are not perfectly separated by price range.
    It should mean that another parameter has been impacting the clustering.
    """

    # Plot PCA to explain the clusters
    plot_PCA(X)
    """
    We can see that the first component explains more than 90% of the variance in 
    the data and the second component less than 10%.
    Now, let's see which dimension is impacting the most the variance in the data.
    """

    # Plot impact of each dimension on the PCA
    plot_dimension_impact(X)
    """
    We can see that all dimensions are impacting the first principal component.
    The first four dimensions are impacting the first principal component the most 
    and they are related to the price of BTC.
    But the fifth dimension, the volume, is also impacting the first principal component, 
    which could explain the fact that the clusters are not prefectly separated by price range.
    """

    # Plot KMeans with 5 clusters using price and volume as y axis
    plot_clusters_price_volume(X)
    """
    We can see that the points are clustered by price because the cluster are almost
    perfectly separated in groups of volume range.
    We can also see that the volume is not impacting much the clustering since the
    clusters look quite randonm when plotted with the volume as y axis.
    """
    
    # Conclusion
    """
    In conclusion, we can say that this clustering has not been useful because the clusters were
    only separating different price ranges.
    It can be explained by the fact that after the normalization, the price was the variable with
    the highest variance.
    Next time, we will use the variation of the price instead of the price itself.
    """

if __name__ == "__main__":
    main()