import numpy as np
import eigen_features
from sklearn.neighbors import NearestNeighbors


def get_eigen_values_with_radius(nbr_arr, knn_arr, no_rand_pts):
    """
      This function generates the eigen values for neighbors within a radius of all the points in in_file_name if
      no_rand_pts = 0. If no_rand_pts > 0, the specified number of random points are chosen from points in in_file_name.
      The neighbors are generated using the points in nbr_data_file_name.
      The following radii are used to get the neighbors: 0.1, 0.25, 0.5, 0.75 and 1 m. Since the number of neighbors in 1 m
      radius can be quite high in a dense pointcloud, to avoid memory error it is advised to split a plot into multiple subplots
      and given the whole plot file as nbr_data_file_name and individual subplot files as in_file_name.

       Parameters
       ----------
       nbr_arr: numpy array
           Three-dimensional point cloud of a bigger area to construct the neighborhood data
       knn_arr: numpy array
           Three-dimensional point cloud of a subset of points from nbr_arr for which the eigen features are calculated
       no_rand_pts:int
            Number of points to randomly select from knn_arr for which the eigen features are calculated. If 0, features are calculated on
            all points of knn_arr

        Returns
        -------
        feature_arr: arr
            Array of size (m x 18) where m is the number of points and out of 18, 3 fields correspond to x,y,z dimension of points and the
            other 15 come from the 3 eigen values corresponding to x, y and z dimension for 5 different radii of the pointcloud

       """


    rand_int = np.random.choice(np.shape(knn_arr)[0], int(no_rand_pts), replace=False)

    if rand_int > 0:
        knn_arr = knn_arr[rand_int,:]

    #print('Generating neighbor tree')
    nbrs = NearestNeighbors(metric='euclidean',algorithm='kd_tree', leaf_size=15, n_jobs=-1).fit(nbr_arr[:, 0:3])

    print(nbr_arr.shape, knn_arr.shape)
    rad = [0.1, 0.25, 0.5, 0.75, 1.0]
    full_eigen_ratios = []
    
    for i in range(0, len(rad)):
        #print(rad[i])
        eigen_ratios = eigen_features.get_vectorized_eigen_vals(nbrs,  nbr_arr[:,0:3], knn_arr[:,0:3], rad = rad[i], ratios=True)

        if i == 0:
            full_eigen_ratios = eigen_ratios
        else:
            full_eigen_ratios = np.column_stack((full_eigen_ratios, eigen_ratios))
            #print(np.shape(full_eigen_ratios))
        

    data_with_eigen_features = np.column_stack((knn_arr[:,0:3], full_eigen_ratios, knn_arr[:,-1]))

    return data_with_eigen_features

