import numpy as np
import pandas as pd
import h5py
from scipy.spatial.distance import cdist
import time

class Retriever(object):
    def __init__(self):
        dir = 'imagenet2010_fc7'
        #dir = 'py_vgg16_fc7'
        self.qf = h5py.File('db/'+dir+'/db_feat4096.h5')
        self.nf = h5py.File('db/'+dir+'/db_filename.h5')
        self.queryDB = np.array(self.qf['feat_db'])
        self.path = np.array(self.nf['list_im']).flatten()
        if 'cata_im' in self.nf:
            self.cata = np.array(self.nf['cata_im']).flatten()
        else:
            self.cata = []
    
    def retrieval(self, img_name):
        tic = time.time()

        queryImg = pd.read_csv('%s.csv'%img_name,header=None).as_matrix()
    
        #print queryDB, queryDB.shape
        #print queryImg, queryImg.shape
        dists = cdist(queryImg, self.queryDB, 'euclidean')
        ind = np.argsort(dists).flatten()
    
        ##print dists, ind, path, path.shape
        k = 100
        ret = [self.path[i] for i in ind[:k]]
        if len(self.cata) > 0:
            cata = [self.cata[i] for i in ind[:k]]
        else:
            cata = ['' for i in ind[:k]]
        
        print '---- Retieval for %f s ----'%(time.time()-tic)
        #print ret
        return ret, cata

