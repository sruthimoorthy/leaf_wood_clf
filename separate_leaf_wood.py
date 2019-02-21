import numpy as np
import generate_features
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import pickle
from sklearn.cluster import KMeans
import glob

def pred(model_name, X_test, out_file):

    X_feat = X_test[:,3:-1]

    X_feat[np.isnan(X_feat)] = 0

    clf = pickle.load(open(model_name, 'rb'))
    y_pred = clf.predict(X_feat)
    np.savetxt(out_file, np.column_stack((X_test[:,0:3], y_pred)), fmt='%1.3f')

def separate_leaf_wood(in_file, model_file, out_file):
    data = np.loadtxt(in_file)
    data_with_features = generate_features.get_eigen_values_with_radius(data, data, 0)
    pred(model_file, out_file , data_with_features)

in_file = '' #input file name
out_file = '' #output file name to save the predictions.
model_file = '' #download the model from the following link: https://www.dropbox.com/s/dpe8hzxorufv7qt/leaf_vs_wood_clf_model.sav?dl=0

separate_leaf_wood(in_file, model_file, out_file)