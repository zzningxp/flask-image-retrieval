import sys
#sys.path.append("/Users/wszzn/develope/caffe/python")
#sys.path.append("/home/cxy/software/caffe-cvprw15-master/python")
#sys.path.append("/home/cxy/software/caffe-cvprw15-master/python/caffe")

sys.path.append("/home/zwx/caffe/python")
sys.path.append("/home/zwx/caffe/python/caffe")

import caffe

#chenxinyaun 
caffe.set_mode_cpu()
#caffe._caffe.set_mode_cpu()

import numpy as np
import pandas as pd
import time

class CaffeNet(object):
    def __init__(self):
        self.netinit()
        
    def netinit(self):
        deffile = 'models/VGG16/deploy_l7.prototxt'
        modfile = 'models/VGG16/VGG_ILSVRC_16_layers.caffemodel'
        self.net = caffe.Net(deffile,modfile,caffe.TEST)
        
    def feature_exact(self, img_name):
        tic = time.time()
    
        transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        transformer.set_mean('data', np.load('db/ilsvrc_2012_mean.npy').mean(1).mean(1))
        transformer.set_transpose('data', (2,0,1))
        transformer.set_channel_swap('data', (2,1,0))
        transformer.set_raw_scale('data', 255.0)
        
        #note we can change the batch size on-the-fly
        #since we classify only one image, we change batch size from 10 to 1
        self.net.blobs['data'].reshape(1,3,224,224)
        
        #load the image in the data layer
        im = caffe.io.load_image(img_name)
        self.net.blobs['data'].data[...] = transformer.preprocess('data', im)
        
        #compute
        out = self.net.forward()
        
        # other possibility : out = self.net.forward_all(data=np.asarray([transformer.preprocess('data', im)]))
        
        #predicted predicted class
        ret = self.posthandler(out['fc7'])
        #print ret, ret.shape
        pd.DataFrame(ret).to_csv('%s.csv'%img_name, index=False, header=False)

        print '---- Feature exacted for %f s ----'%(time.time()-tic)
        return True
    
    def posthandler(self, arr):
        return arr
