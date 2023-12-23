from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import numpy as np

def plot_BTC_USD(data):
    # Compute average price per day
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
    # Compute the clusters
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
    pca = PCA().fit(data)

    plt.bar(
        range(1, len(pca.explained_variance_ratio_)+1),
        pca.explained_variance_ratio_
        )
    plt.title("PCA explained variance ratio")
    plt.show()

def plot_dimension_impact(data):
    pca = PCA().fit(data)

    plt.bar(
        range(1, len(pca.components_)+1),
        pca.components_[0]
        )
    plt.title("PCA dimension impact")
    plt.show()

def plot_clusters_volume(data):
    kmeans_5 = KMeans(n_clusters=5, random_state=0, n_init=10).fit(data)

    plt.scatter(
        [a for a in range(len(data))], 
        data[:,4], 
        c=kmeans_5.labels_,
        cmap='rainbow', s=1.5
        )
    plt.title("KMeans 5 clusters")
    plt.show()

def main():
    # Load data
    """
    The data is from Yahoo Finance and is the daily price of BTC-USD.
    The columns are: Date, Open, High, Low, Close, Adj Close, Volume.
    We don't keep the date column and the adjusted close column.
    """
    X = np.loadtxt("BTC-USD_daily.csv", delimiter=",", skiprows=1, usecols=(1,2,3,4,6))

    # Plot BTC-USD in linear and log scale
    plot_BTC_USD(X)

    # Plot KMeans with 2 and 5 clusters
    plot_clusters_kmeans(X)
    """
    The clusters seem to be quite random. We will have to dig deeper to understand what is happening.
    """

    # Plot PCA to explain the clusters
    plot_PCA(X)
    """
    We can see that the first component explains more than 99% of the variance in the data.
    Now, let's see which dimension is impacting the most the variance in the data.
    """

    # Plot impact of each dimension on the PCA
    plot_dimension_impact(X)
    """
    We can see that the fifth dimension is the one impacting the most the first principal component, hence almost all the variance in the data.
    This dimension is the volume of BTC exchanged per day.
    To confirm this, let's plot the clusters using the volume as y axis.
    """

    # Plot KMeans with 5 clusters using volume as y axis
    plot_clusters_volume(X)
    """
    We can see that the points are clustered by volume because the cluster are perfectly separated in groups of volume range.
    As expected, the volume is the most important feature to explain the clusters.
    """
    
    # Conclusion
    """
    In conclusion, we can say that this clustering has not been useful because the clusters were only separating different price ranges.
    This happened because the volume is the variable which varies the most in absolute value.
    This is why we will have to normalize the data before clustering.
    """

if __name__ == "__main__":
    main()