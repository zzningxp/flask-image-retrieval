from context import *
import unittest
import numpy as np
import pandas as pd

class TestImageRetrieval(unittest.TestCase):
    def testRetrieval(self):
        print '+++++++'
        fn = 'test/test.jpg'
        retri = image_retrieval.Retriever()
        retri.retrieval(fn)

if __name__ == '__main__':
    unittest.main()
