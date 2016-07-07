from context import *
import unittest
import numpy as np
import pandas as pd

class TestImageCaffe(unittest.TestCase):
    def testCaffe(self):
        print '++++'
        fn = 'test/test.jpg'
        caffenet = image_caffe.CaffeNet()
        caffenet.feature_exact(fn)
        res = pd.read_csv('%s.csv'%fn, header=None).as_matrix()
        print res,res.shape

if __name__ == '__main__':
    unittest.main()
