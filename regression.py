# 用不同方法实现多元线性回归

import pandas as pd
import numpy as np
from sklearn import datasets
import warnings
warnings.filterwarnings('ignore')


# 用最小二乘法实现
class LinearRegression1:

    def __init__(self):
        X = datasets.load_boston().data
        y = datasets.load_boston().target
        self.X = X
        self.y = y


    def fit(self):
        self.X = np.insert(self.X,0,1,axis=1)
        self.coef_ = np.linalg.inv(self.X.T.dot(self.X)).dot(self.X.T).dot(self.y)
        # self.intercept_ = np.sum(self.y - self.X.dot(self.coef_))
        return self.coef_
            # , self.intercept_


    def predict(self, x):
        a = self.fit().dot(x)
        return a.reshape(-1,1)
