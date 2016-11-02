import numpy as np
import pandas as pd
import h5py
from scipy.spatial.distance import cdist
from sklearn import preprocessing
import time

class Retriever(object):
    def __init__(self):
        #dir = 'imagenet2010_fc7'
        dir = 'py_vgg16_fc7'
        self.qf = h5py.File('db/'+dir+'/db_feat4096.h5')
        self.nf = h5py.File('db/'+dir+'/db_filename.h5')
        self.queryDB = np.array(self.qf['feat_db'])
        self.signDB = np.sign(self.queryDB)
        self.path = np.array(self.nf['list_im']).flatten()
        if 'cate_im' in self.nf:
            self.cate = np.array(self.nf['cate_im']).flatten()
        else:
            self.cate = []
    
    def euclidean_dist(self, queryImg):
        queryImg = preprocessing.normalize(queryImg, norm='l2')
        dists = cdist(queryImg, self.queryDB, 'euclidean')
        return np.argsort(dists).flatten()

    def hamming_dist(self, queryImg):
        q = np.sign(queryImg)
        res = []
        for i, x in enumerate(self.signDB):
            res.append((i, np.count_nonzero(x==q)))
        res.sort(key=lambda tup: tup[1], reverse=True)
        ind = [x[0] for x in res]
        return ind

    def retrieval(self, img_name):
        tic = time.time()

        queryImg = pd.read_csv('%s.csv'%img_name,header=None).as_matrix()
    
        #ind = self.euclidean_dist(queryImg)
        ind = self.hamming_dist(queryImg)
    
        ##print dists, ind, path, path.shape
        k = 100
        ret = [self.path[i] for i in ind[:k]]
        if len(self.cate) > 0:
            cate = [self.cate[i] for i in ind[:k]]
        else:
            cate = ['' for i in ind[:k]]
        
        print '---- Retieval for %f s ----'%(time.time()-tic)
        #print ret
        return ret, cate

