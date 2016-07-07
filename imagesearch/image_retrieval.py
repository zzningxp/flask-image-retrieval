import numpy as np
import pandas as pd
import h5py
from scipy.spatial.distance import cdist
import time

class Retriever(object):
    def __init__(self):
        self.f = h5py.File('db/vgg16_fc7/db_feat4096.mat')
        print self.f.keys()
        self.queryDB = np.array(self.f['feat_db'])
        self.path = np.array(self.f['list_im']).flatten()
    
    def retrieval(self, img_name):
        tic = time.time()

        queryImg = pd.read_csv('%s.csv'%img_name,header=None).as_matrix()
    
        #print queryDB, queryDB.shape
        #print queryImg, queryImg.shape
        dists = cdist(queryImg, self.queryDB, 'euclidean')
        ind = np.argsort(dists).flatten()
    
        ##print dists, ind, path, path.shape
        k = 30
        ret = [u''.join(unichr(c) for c in self.f[self.path[i]]).replace('/work/project/my-work/image_retrieval/dataset/','') for i in ind[:k]]

        print '---- Retieval for %f s ----'%(time.time()-tic)
        return ret

