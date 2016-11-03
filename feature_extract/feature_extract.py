# -*- coding: UTF-8 -*-
import numpy as np
#import matplotlib.pyplot as plt
import os
import sys
import pickle
import struct
import h5py
import pandas as pd
#import sys,cv2
sys.path.append("/home/zwx/caffe/python")
sys.path.append("home/zwx/caffe/python/caffe")
import caffe
caffe.set_mode_cpu()
# 运行模型的prototxt
deployPrototxt = 'models/VGG16/deploy_l7.prototxt'
# 相应载入的modelfile
modelFile = 'models/VGG16/VGG_ILSVRC_16_layers.caffemodel'
meanFile = 'db/ilsvrc_2012_mean.npy'
# 需要提取的图像列表
imageListFile = '/home/cxy/code/images/images_total.txt'
imageBasePath = '/home/cxy/code/images'
image_cateFile='/home/cxy/code/images/image_cate.txt'
postfix = '.csv'

# 初始化函数的相关操作
def initilize():
    print 'initilize ... '

    net = caffe.Net(deployPrototxt, modelFile,caffe.TEST)
    return net  
# 提取特征并保存为相应地文件
def extractFeature(imageList, net):
    # 对输入数据做相应地调整如通道、尺寸等等
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data', np.load(meanFile).mean(1).mean(1)) # mean pixel
    transformer.set_raw_scale('data', 255)  
    transformer.set_channel_swap('data', (2,1,0))  
    # set net to batch size of 1 如果图片较多就设置合适的batchsize 
    net.blobs['data'].reshape(1,3,224,224)      #这里根据需要设定，如果网络中不一致，需要调整
    num=0
   # fea_file = '/home/cxy/code/images/feature.txt'
    resarray = []
    file=h5py.File('feature_db.h5','w')
    index=0
    for imagefile in imageList:
        imagefile_abs = os.path.join(imageBasePath, imagefile)
        print imagefile_abs
        net.blobs['data'].data[...] = transformer.preprocess('data', caffe.io.load_image(imagefile_abs))
        out = net.forward()
	print out['fc7']
	resarray.append((index,out['fc7']))
	index+=1
    file.create_dataset('feat_db',data=resarray)
    file.close()
        #fea_file = imagefile_abs.replace('.jpg',postfix)
#        num +=1
#        print 'Num ',num,' extract feature ',fea_file
 #       with  open(fea_file,'wb') as f:
  #          for x in xrange(0, net.blobs['fc7'].data.shape[0]):
   #             for y in xrange(0, net.blobs['fc7'].data.shape[1]):
    #                f.write(struct.pack('f', net.blobs['fc7'].data[x,y]))
#	temp=pd.DataFrame(out['fc7']).to_csv('%s.csv'%imagefile, index=False, header=False)
	
#	fea_file = imagefile_abs.replace('.jpg.csv',postfix)
 #   return fea_file
# 读取文件列表
def readImageList(imageListFile):
    imageList = []
    with open(imageListFile,'r') as fi:
        while(True):
            line = fi.readline().strip().split()# every line is a image file name
            if not line:
                break
            imageList.append(line[0]) 
    print 'read imageList done image num ', len(imageList)
    return imageList

#封装成hdf5库文件
def hdf5DB(name_file,cate_file):
  #  file=h5py.File('feature_db.h5','w')
  #  queryImg = pd.read_csv(feature,header=None).as_matrix()
  #  file.create_dataset('feat_db',data=queryImg)
    file=h5py.File('name_db.h5','w')
    file.create_dataset('list_im',data=name_file)
    file.create_dataset('cate_im',data=cate_file)





if __name__ == "__main__":
    net = initilize()
    imageList = readImageList(imageListFile) 
    feature=extractFeature(imageList, net)
    hdf5DB(imageListFile,image_cateFile)
