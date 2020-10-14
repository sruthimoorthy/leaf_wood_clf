import numpy as np
import eigen_val_and_vec_features
from sklearn.neighbors import NearestNeighbors
import math

def generate_only_eigen_ratios_with_radius(nbr_pts, pt):
   
    nbrs = NearestNeighbors(metric='euclidean',algorithm='kd_tree', leaf_size=15, n_jobs=-1).fit(nbr_pts[:, 0:3])
    print(nbr_pts.shape, pt.shape)
    rad = [0.1, 0.25, 0.5, 0.75, 1]
    full_eigen_ratios = []
    for i in range(0, len(rad)):
        print(rad[i])
        eigen_ratios = eigen_val_and_vec_features.get_vectorized_eigen_vals(nbrs,  nbr_pts[:,0:3], pt[:,0:3], rad = rad[i], ratios=True)

        if i == 0:
            full_eigen_ratios = eigen_ratios
        else:
            full_eigen_ratios = np.column_stack((full_eigen_ratios, eigen_ratios))
            print(np.shape(full_eigen_ratios))
    print(full_eigen_ratios.shape, pt[:,0:3].shape)
    full_eigen_ratios = np.column_stack((pt[:,0:3],full_eigen_ratios))
    training_data = np.column_stack((full_eigen_ratios[:,0:3], full_eigen_ratios[:,3:6], full_eigen_ratios[:,9:12],full_eigen_ratios[:,15:18],full_eigen_ratios[:,21:24], full_eigen_ratios[:,27:30],full_eigen_ratios[:,6:9], full_eigen_ratios[:,12:15],full_eigen_ratios[:,18:21],full_eigen_ratios[:,24:27], full_eigen_ratios[:,30:]))

    return training_data


def get_all_features(knn_arr, in_file):
    % Change the number of points based on the memory available
    no_pts_in_loop = 200000 
    data_with_feat = []
    pts_count = knn_arr.shape
    print(pts_count[0])
    if pts_count[0] > no_pts_in_loop:
        loop_count = int(math.ceil(pts_count[0]/no_pts_in_loop))
        print(loop_count)
        start_idx = 0
        end_idx = no_pts_in_loop
        for j in range(0,loop_count):
            print(start_idx, end_idx)
            data_with_feat_temp = generate_only_eigen_ratios_with_radius(knn_arr, knn_arr[start_idx:end_idx,:])
            if j == 0:
                data_with_feat = data_with_feat_temp
            else:
                data_with_feat = np.row_stack((data_with_feat, data_with_feat_temp))
            if pts_count[0] < end_idx + no_pts_in_loop:
                start_idx = end_idx
                end_idx = pts_count[0]
            else:
                start_idx = end_idx
                end_idx = end_idx + no_pts_in_loop
    else:
        data_with_feat = generate_only_eigen_ratios_with_radius(knn_arr, knn_arr)
    print(data_with_feat.shape)
    np.savetxt(in_file+'_feat.txt', data_with_feat, fmt='%1.4f')
    return data_with_feat


