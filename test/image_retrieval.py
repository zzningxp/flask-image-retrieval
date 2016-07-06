from context import *
import unittest
import numpy as np
import pandas as pd

class TestImageRetrieval(unittest.TestCase):
    def testRetrieval(self):
        fn = 'test/test.jpg'
        image_retrieval.retrieval(fn)

if __name__ == '__main__':
    unittest.main()
