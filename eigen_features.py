import numpy as np

array_to_find_nbrs = []

def get_elements(idx):
    return array_to_find_nbrs[idx,:]

def get_covariance(arr):
    return np.cov(np.asarray(arr).T)

def get_eigen_vals(cov_arr):
    evals = np.linalg.svd(cov_arr, compute_uv=False)
    eval_ratio = evals/np.sum(evals)
    return eval_ratio

def get_vectorized_eigen_vals(nbrs, nbr_arr , knn_arr, rad, ratios = True):
    """
        Function to calculate eigen values for the neighbors within a given radius of all the points in knn_arr

        Parameters
        ----------
        nbrs: Nearest neigbor classifier
        nbr_arr: numpy array
           Three-dimensional point cloud of a bigger area to construct the neighborhood data
        knn_arr: numpy array
           Three-dimensional point cloud of a subset of points from nbr_arr for which the eigen features are calculated
        rad: float
            Limiting distance of neighbors to return.

        Returns
        -------
        eval_ratio: arr
            Array of size (m x 3) where m is the number of points and 3 refers to 3 eigen values corresponding to x, y and z dimension of
            the pointcloud

        """

    #print('Vectorized eigen val')
    nbr_idx = nbrs.radius_neighbors(knn_arr, radius=rad, return_distance=False)
    global array_to_find_nbrs
    array_to_find_nbrs = np.asarray(nbr_arr)
    
    #print('Getting elements from index')
    
    get_elem_func = np.vectorize(get_elements, otypes=[np.object])
    arr_stack = get_elem_func(nbr_idx)
    #print('Calculating covariance matrix')
    
    cov_func = np.vectorize(get_covariance, otypes=[np.object])
    cov_mat = cov_func(arr_stack)
    
    #print('Calculating eigen values of the covariance matrix')
    
    eval_fun = np.vectorize(get_eigen_vals, otypes=[np.object])
    eval_ratio = eval_fun(cov_mat)
    #print(eval_ratio)
    return np.row_stack((eval_ratio))
