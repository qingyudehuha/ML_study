# 用不同方法实现多元线性回归

import pandas as pd
import numpy as np
from sklearn import datasets
import warnings
warnings.filterwarnings('ignore')



# 最小二乘法实现线性回归

class LinearRegressionLSM:

    def __init__(self):
        X = datasets.load_boston().data
        y = datasets.load_boston().target
        self.X = X
        self.y = y


    def fit(self):
        self.X = np.insert(self.X,0,1,axis=1)
        self.coef_ = np.linalg.inv(self.X.T.dot(self.X)).dot(self.X.T).dot(self.y)
        # self.intercept_ = np.average(self.y) - self.coef_.dot(np.average(self.X,axis=0))
        return self.coef_
            # , self.intercept_


    def predict(self, x):
        a = self.fit().dot(x)
        return a.reshape(-1,1)
    
    
    
    
# 用牛顿法实现
