import numpy as np
import pylab as pl
import matplotlib.pyplot as plt

def process_data(name):
    """
    Returns the processed dataset along with its radius.
    Args:
        name: string, should be '20' or '21'. Determines the dataset.
    """
    # load data from csv file
    test_data = np.loadtxt('check'+name+'.csv')

    # initialize the data
    points = test_data[:, 0:-1].copy()
    labels = test_data[:, -1:].copy()
    Y = labels.ravel()

    # get radius of dataset
    radius = 0
    for point in points:
        norm = np.linalg.norm(point)
        if norm > radius:
            radius = norm

    return points, labels, Y, radius

def plot_points(X, Y, title = "", margin=1):
    """
    2D scatterplot of the input dataset.
    Args:
        X: A matrix where each row corresponds to a data point.
        Y: The label of each point (1 or -1)
        title: string, title of the plot
        margin: float, determines spacing of plot
    """
    x_min, x_max = X[:, 0].min() - margin, X[:, 0].max() + margin
    y_min, y_max = X[:, 1].min() - margin, X[:, 1].max() + margin
    plt.scatter(X[:, 0], X[:, 1], c=(1.-Y).ravel(), s=50, cmap = pl.cm.cool)
    plt.axis([x_min,x_max,y_min,y_max])
    plt.title(title)
    plt.show()
