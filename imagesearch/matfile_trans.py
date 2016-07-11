import numpy as np
import pandas as pd
import h5py
from scipy.spatial.distance import cdist
from sklearn import preprocessing
import time

def transform():
    #rf = h5py.File('db/vgg16_fc7/db_feat4096.mat')
    rf = h5py.File('db/imagenet2010_mat/imagenet_feat4096.mat')
    print rf.keys()
    get_feature_db(rf)
    #get_path_db_from_mat(rf)

def get_feature_db(rf):
    qf = h5py.File('db/imagenet2010_fc7/db_feat4096.h5', 'w')
    queryDB = np.array(rf['feat'])
    queryDB = preprocessing.normalize(queryDB, norm='l2')
    qf.create_dataset('feat_db', data=queryDB)
    print queryDB.shape

def get_path_db_from_mat(rf):
    nf = h5py.File('db/imagenet2010_fc7/db_filename.h5', 'w')
    path = np.array(rf['list_im']).flatten()
    retpath = []
    for i,p in enumerate(path):
        try:
            st = u''.join(unichr(c) for c in rf[p]).replace('/work/project/my-work/image_retrieval/dataset/','')
        except:
            st = 'unknown'
        st = str(st)
        retpath.append(st)
        if i % 1000 == 0:
            print i

    nf.create_dataset('list_im', data=retpath)

def get_path_db_from_txt(fname):
    name = np.loadtxt(fname, dtype='str')
    cata = [i.split('/')[1] for i in name]

    nf = h5py.File('db/imagenet2010_fc7/db_filename.h5', 'w')
    nf.create_dataset('list_im', data=name)
    nf.create_dataset('cate_im', data=cata)

if __name__ == '__main__':
    transform()
    #get_path_db_from_txt('db/imagenet2010_fc7/train_rel_name.txt')
    
