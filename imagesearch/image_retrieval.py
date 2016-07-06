import numpy as np
import pandas as pd
import h5py
from scipy.spatial.distance import cdist

def retrieval(img_name):
    f = h5py.File('db/vgg16_fc7/db_feat4096.mat')
    print f.keys()
    queryDB = f['feat_db']
    queryDB = np.array(queryDB)

    queryImg = pd.read_csv('%s.csv'%img_name,header=None).as_matrix()

    #print queryDB, queryDB.shape
    #print queryImg, queryImg.shape
    dists = cdist(queryImg, queryDB, 'euclidean')
    ind = np.argsort(dists).flatten()
    path = np.array(f['list_im']).flatten()

    ##print dists, ind, path, path.shape
    k = 10
    ret = [u''.join(unichr(c) for c in f[path[i]]).replace('/work/project/my-work/image_retrieval/dataset/','') for i in ind[:k]]
    return ret

