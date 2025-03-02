#2D visualization of high dimensional data.
import numpy as np
import utils
from sklearn import manifold

#Parameters
name = '21'
out_dim = 2 #Dimension for t-sne

#Load pre-processed data
points, labels, _, radius = utils.process_data(name)

#Run t-sne
rng = np.random.RandomState(0)
t_sne = manifold.TSNE(
    out_dim,
    learning_rate="auto",
    perplexity=100,
    n_iter=1000,
    n_iter_without_progress=500,
    init="pca",
    random_state=rng,
    )
data_t_sne = t_sne.fit_transform(points)

utils.plot_points(data_t_sne, labels, "Visualization of Dataset", 0.2)
