import numpy as np
import pickle
import prepare_data_for_prediction



def separate_leaf_wood(in_file, model_file, out_file):

    data = np.loadtxt(in_file)
    data_with_features = prepare_data_for_prediction.get_all_features(data, in_file)
    data_feat = data_with_features[:,3:]

    data_feat[np.isnan(data_feat)] = 0

    clf = pickle.load(open(model_file, 'rb'))
    y_pred = clf.predict(data_feat)
    np.savetxt(out_file, np.column_stack((data[:,0:3], y_pred)), fmt='%1.3f')
    



in_file = '' #input file name. Expects a ".txt" file with xyz coordinates of points separated by space. The file can have more than just xyz values for points (eg. reflectance). However, the first three attributes for every point should be the xyz coordinates. 
out_file = '' #output file name to save the predictions.
model_file = 'leaf_wood_RF_final_model.sav' #download the model from the following link: https://www.dropbox.com/s/dpe8hzxorufv7qt/leaf_vs_wood_clf_model.sav?dl=0

separate_leaf_wood(in_file, model_file, out_file)

