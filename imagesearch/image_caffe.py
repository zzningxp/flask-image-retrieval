import sys
sys.path.append("/Users/wszzn/develope/caffe/python")
import caffe
caffe.set_mode_cpu()
import numpy as np
import pandas as pd

def feature_exact(img_name):
    deffile = 'models/VGG16/deploy_l7.prototxt'
    modfile = 'models/VGG16/VGG_ILSVRC_16_layers.caffemodel'
    net = caffe.Net(deffile,modfile,caffe.TEST)
    
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_mean('data', np.load('db/ilsvrc_2012_mean.npy').mean(1).mean(1))
    transformer.set_transpose('data', (2,0,1))
    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data', 255.0)
    
    #note we can change the batch size on-the-fly
    #since we classify only one image, we change batch size from 10 to 1
    net.blobs['data'].reshape(1,3,224,224)
    
    #load the image in the data layer
    im = caffe.io.load_image(img_name)
    net.blobs['data'].data[...] = transformer.preprocess('data', im)
    
    #compute
    out = net.forward()
    
    # other possibility : out = net.forward_all(data=np.asarray([transformer.preprocess('data', im)]))
    
    #predicted predicted class
    ret = posthandler(out['fc7'])
    #print ret, ret.shape
    pd.DataFrame(ret).to_csv('%s.csv'%img_name, index=False, header=False)
    return True

def posthandler(arr):
    return arr
