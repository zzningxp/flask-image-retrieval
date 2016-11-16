# -*- coding: UTF-8 -*-
import numpy as np
import h5py
import pandas as pd
import  image_caffe
import  sys
import  os
# 需要提取的图像列表
imageListFile = 'data/images_total_steel.txt'
imageBasePath = 'data/images'
image_cateFile='data/image_cate.txt'
# 提取特征并保存为相应地文件
def extractFeature(imageList, net):
  
    file=h5py.File('feature_db.h5','w')
    resarray = []
    for imagefile in imageList:
        imagefile_abs = os.path.join(imageBasePath, imagefile)
	caffenet.feature_exact(imagefile_abs)
        res = pd.read_csv('%s.csv'%imagefile_abs, header=None).as_matrix()
#       print res,res.shape
	resarray.append(res)

    file.create_dataset('feat_db',data=resarray)	

    file.close()



def readList(ListFile):
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

    file=h5py.File('name_db.h5','w')
    file.create_dataset('list_im',data=name_file)
    file.create_dataset('cate_im',data=cate_file)
    file.close()




if __name__ == "__main__":
   
    imageList = readList(imageListFile)
    cateList = readList(image_cateFile)
    caffenet = image_caffe.CaffeNet() 
    extractFeature(imageList, caffenet)
    hdf5DB(imageList,cateList)
