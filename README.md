#flask-image-retrieval
**keywords** : `CNN`, `feature exact`, `image retrieval`, `caffe`

-------------------------------------------
Python flask web service image retrieval project based https://github.com/kevinlin311tw/caffe-cvprw15 Matlab version.
## Introduction
**Flask image retrieval** is an image retrieval framework based on deep  convolutional networks implemented by python , it mainly includes image feature extract part (to form the feature library )and feature retrieval part( search the image accoding to similarity).
This code has been tested on `ubuntu 14.04 LTS` and `python 2.7`.

##Install Caffe
The Caffe version in this README is from `anaconda`, and it is configured to a `no-gpu version`. However, it has also been tested on GPU version.

###Install anaconda
Download the `anaconda` from https://www.continuum.io/downloads, select the `python2.7` version. run to install anaconda
```bash
$ ./Anaconda.sh
```
When install anaconda, many dependencies like below  will be installed automatically:
  - `flask`:  Alightweight web application framework, this project is based on this web framework.
  - `pandas`: A data analysis package for Python
  - `qt` ,`numpy`,`matplotlib` and so on
###Install dependencies
```bash
 $ conda install  -c conda-forge caffe
```
Use command below to install several dependencies automatically:
  - `opencv`: open source computer vision library
  - `lmdb`, `leveldb`,`snappy`:IO libraries (leveldb requires *snappy*)
  -  `openBLAS`: basic linear algebra subprograms, as the ackend of matrix and vector computations
  - `Boost` : c++ extension library
  - `protobuf`, `glog`, `gflags`  and so on
###Configure Caffe
Configure the build by copying and modifying the example `Makefile.config` for your setup. Uncomment the relevant lines accoding to your anaconda python.
Uncomment to build without `GPU` support
```bash
CPU_ONLY := 1
```
If you're using `OpenCV 3`, configure as below
```bash
OPENCV_VERSION := 3
OPENCV_INCLUDE := /Your/Anaconda/Path/pkgs/opencv3-3.1.0-py27_0/include
OPENCV_LIB := /Your/Anaconda/Path/pkgs/opencv3-3.1.0-py27_0/lib
```
BLAS choice: atlas for ATLAS (default) , mkl for MKL, `open` for `OpenBlas`
```bash
BLAS := open 
BLAS_INCLUDE := /Your/Anaconda/Path/pkgs/openblas-0.2.19/include
BLAS_LIB := /Your/Anaconda/Path/pkgs/openblas-0.2.19/lib
```

```bash
PYTHON_VERSION := 2.7
ANACONDA_HOME := /Your/Anaconda/Path
PYTHON_INCLUDE := $(ANACONDA_HOME))/include \
              `$(ANACONDA_HOME)/include/python2.7 \
               `$(ANACONDA_HOME)/lib/python2.7/site-packages/numpy/core/include \
PYTHON_LIB := $(ANACONDA_HOME)/lib
```
###Build caffe
Go to https://github.com/BVLC/caffe, download zip archive and unpack it. Or clone the source code. Enter caffe-home/python directory to install Python packages:
```bash
for req in $(cat requirements.txt); do conda install $req;
```
Then enter caffe-home/python directory:
```bash
$ make all 
$ make runtest
$ make pycaffe
```

Verify the installation by running `python -c "import caffe;print caffe.__version__"`


## Build your own feature library
- You can download deploy file from [vgg_train_val.prototxt](http://cs.stanford.edu/people/karpathy/vgg_train_val.prototxt) and 
- Download your own dataset and train your  model, it will generate `filename.caffemodel`file.
- In `imagesearch` dictionary, there is a `feature_extract.py` file . You can set as below :

 ```
 deployPrototxt = '/Your/Deploy/Path/XXX.prototxt';
 modelFile = '/Your/Model/Path/XXX.caffemodel';
 meanFile = 'db/XXX_mean.npy';
 imageListFile = '/Your/Image/Path/images_total.txt';
 imageBasePath = '/Your/Image/Path';
 image_cateFile='/Your/Image/Path/image_cate.txt';
```

sudo pip install simplejson
sudo pip install flask
sudo pip install pillow
sudo pip install scikit-image

- Run `feature_extract.py` and generate feature library file `feature_db.h5`.

##Run the demo
- Run `feature_extract.py` to extract feature. the feature library will be saved in `hdf5` format.
- Run `app.py` to make a link with browser.
- Open a browser, enter `YourIpAdder:5000` in the address bar. 
- Load your image and then it will show retrieval result automatically.


