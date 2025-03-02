#Create a dataset of points in 2 distinct classes.
import numpy as np
import activations

def make_three_rings_ND(a,b,c,dim,file_name):
    """
    Creates an n-dimensional dataset of 3 concentric hyperspheres
    with alternating classes.
    Args:
        a: Number of points in the innermost hypersphere. These points will
        be in the class +1.
        b: Number of points in the middle hypersphere. These points will
        be in the class -1.
        c: Number of points in the outer hypersphere. These points will
        be in the class +1.
        dim: Dimension of the dataset.
        fileName: Name of the dataset file to be created.
    """    
    data = np.empty((a+b+c,dim+1))
    for i in range(a):
        r_first = 100 + 20*np.random.rand()
        vec = r_first*activations.uniform_sphere_sample(dim) #sample randomly from uniform distribution on sphere
        datapoint = np.concatenate((vec,np.array([1])))
        data[i] = datapoint
    for i in range(a,a+b):
        r_second = 200 + 40*np.random.rand()
        vec = r_second*activations.uniform_sphere_sample(dim)
        datapoint = np.concatenate((vec,np.array([-1])))
        data[i] = datapoint
    for i in range(a+b,a+b+c):
        r_third = 300 + 60*np.random.rand()
        vec = r_third*activations.uniform_sphere_sample(dim)
        datapoint = np.concatenate((vec,np.array([1])))
        data[i] = datapoint
    np.savetxt(file_name, data, delimiter=" ")

# make_three_rings_ND(100, 100, 100, 100, "check21.csv")
# make_three_rings_ND(100, 100, 100, 2, "check20.csv")