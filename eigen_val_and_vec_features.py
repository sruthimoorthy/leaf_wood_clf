import numpy as np

array_to_find_nbrs = []

def get_elements(idx):
    return array_to_find_nbrs[idx,:]

def get_covariance(arr):
    return np.cov(np.asarray(arr).T)

def calc_beta_rad(pvec):
    '''
    polar angle [0, pi]
    '''
    return np.arccos(pvec[2])  # arccos:[0, pi]


def calc_gamma_rad(pvec):
    '''
    azimuth angle [0, 2pi]
    '''
    gamma = np.arctan2(pvec[1], pvec[0])
    if gamma < 0.0:
        gamma += 2 * np.pi
    return gamma

def get_eigen_vals(cov_arr):
    cov_arr =  np.nan_to_num(cov_arr, copy=True)
    ev, evals, ev_trans = np.linalg.svd(cov_arr, compute_uv=True)
    zen_angle_0 = calc_beta_rad(ev[:,0])
    azi_angle_0 = calc_gamma_rad(ev[:, 0])
    zen_angle_1 = calc_beta_rad(ev[:, 1])
    azi_angle_1 = calc_gamma_rad(ev[:, 1])
    zen_angle_2 = calc_beta_rad(ev[:, 2])
    azi_angle_2 = calc_gamma_rad(ev[:, 2])
    eval_ratio = evals/np.sum(evals)

    return np.column_stack((eval_ratio[0], eval_ratio[1], eval_ratio[2], zen_angle_0, zen_angle_1 ,zen_angle_2))

def get_vectorized_eigen_vals(nbrs, nbr_arr , knn_arr, rad, ratios = True):
    """
        Function to calculate eigen values and vectors for the neighbors within a given radius of all the points in knn_arr

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
            Array of size (m x 6) where m is the number of points and first 3 columns refers to 3 eigen values corresponding to x, y and z dimension of
            the pointcloud and the next 3 columns are the zenith angle of the 3 eigen vetors

        """

    print('Vectorized eigen val')
    nbr_idx = nbrs.radius_neighbors(knn_arr, radius=rad, return_distance=False)
    global array_to_find_nbrs
    array_to_find_nbrs = np.asarray(nbr_arr)
    
    print('Getting elements from index')
    
    get_elem_func = np.vectorize(get_elements, otypes=[np.object])
    arr_stack = get_elem_func(nbr_idx)
    print('Calculating covariance matrix')
    
    cov_func = np.vectorize(get_covariance, otypes=[np.object])
    cov_mat = cov_func(arr_stack)
    
    print('Calculating eigen values of the covariance matrix')
    
    eval_fun = np.vectorize(get_eigen_vals, otypes=[np.object])
    eval_ratio = eval_fun(cov_mat)
    #print(eval_ratio)
    return np.row_stack((eval_ratio))
