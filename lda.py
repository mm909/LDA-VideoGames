import math
import pandas as pd
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def createCVData(dataset, folds):
    arr = np.arange(dataset.shape[0])
    np.random.shuffle(arr)
    foldsize = int(dataset.shape[0] / folds) + 1
    shuffled = np.array([arr[foldsize*i:foldsize*(i+1)] for i in range(folds)])
    return shuffled

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def predict(x, Batac, Vc):
    return sigmoid(np.dot(np.transpose(Batac[1] - Batac[0]), x) + (Vc[1] - Vc[0]))

class LDA:
    def __init__(self, verbose = 0):
        self.verbose = verbose
        return

    def fitpd(self, data):
        self.data = data
        Class1 = self.data[sales.Publisher == 'Nintendo']
        Class2 = self.data[sales.Publisher == 'Activision']
        self.Xfull = Class1.append(Class2)
        self.n = self.Xfull.shape[0]
        self.d = self.Xfull.shape[1] - 1
        self.y = self.Xfull.Publisher
        self.Nc = self.y.value_counts().values
        self.y = LabelEncoder().fit_transform(self.y.values)
        self.ldaCalc()
        self.printout()
        return

    def fitXy(self, X, y):
        self.X = X
        self.n = self.X.shape[0]
        self.d = self.X.shape[1] - 1
        self.y = y
        self.ldaCalc()
        self.printout()
        return

    def fitXyShared(self, X, y):
        self.X = X
        self.n = self.X.shape[0]
        self.d = self.X.shape[1] - 1
        self.y = y
        self.Nc = np.array([self.n - y.sum(), y.sum()])
        self.PIc = self.Nc / self.n
        self.Uc = np.array([np.mean(self.X[self.y == 0], axis=0), np.mean(self.X[self.y == 1], axis=0)])
        self.SigmaS1 = self.X[self.y == 0] - self.Uc[0]
        self.SigmaSub1 = np.dot(self.SigmaS1.transpose(), self.SigmaS1) / self.Nc[0]
        self.SigmaS2 = self.X[self.y == 1] - self.Uc[1]
        self.SigmaSub2 = np.dot(self.SigmaS2.transpose(), self.SigmaS2) / self.Nc[1]
        self.SigmaHatc = np.array([self.SigmaSub1, self.SigmaSub2])
        self.shared = self.sharedCovMatrix()
        self.Batac = np.array([np.dot(np.linalg.inv(self.shared), self.Uc[0]), np.dot(np.linalg.inv(self.shared), self.Uc[1])])
        self.Vc = np.array([(-1/2) * np.dot(self.Uc[0].transpose(), np.dot(np.linalg.inv(self.shared), self.Uc[0])) + np.log(self.PIc[0]),
                            (-1/2) * np.dot(self.Uc[1].transpose(), np.dot(np.linalg.inv(self.shared), self.Uc[1])) + np.log(self.PIc[1])])
        self.printout()
        return

    def ldaCalc(self):
        self.Nc = np.array([self.n - y.sum(), y.sum()])
        self.PIc = self.Nc / self.n
        self.Uc = np.array([np.mean(self.X[self.y == 0], axis=0), np.mean(self.X[self.y == 1], axis=0)])
        self.SigmaS1 = self.X[self.y == 0] - self.Uc[0]
        self.SigmaSub1 = np.dot(self.SigmaS1.transpose(), self.SigmaS1) / self.Nc[0]
        self.SigmaS2 = self.X[self.y == 1] - self.Uc[1]
        self.SigmaSub2 = np.dot(self.SigmaS2.transpose(), self.SigmaS2) / self.Nc[1]
        self.SigmaHatc = np.array([self.SigmaSub1, self.SigmaSub2])
        self.Batac = np.array([np.dot(np.linalg.inv(self.SigmaHatc[0]), self.Uc[0]), np.dot(np.linalg.inv(self.SigmaHatc[1]), self.Uc[1])])
        self.Vc = np.array([(-1/2) * np.dot(self.Uc[0].transpose(), np.dot(np.linalg.inv(self.SigmaHatc[0]), self.Uc[0])) + np.log(self.PIc[0]),
                            (-1/2) * np.dot(self.Uc[1].transpose(), np.dot(np.linalg.inv(self.SigmaHatc[1]), self.Uc[1])) + np.log(self.PIc[1])])

    def printout(self):
        if self.verbose > 0:
            print('')
            print('X.shape:', self.X.shape)
            print('y.shape:', self.y.shape)
            print('')
            print('n   :', self.n)
            print('Nc  :', self.Nc)
            print('PIc :', self.PIc)
            print('')
            print("Uc       :", self.Uc.shape)
            print("SigmaHatc:", self.SigmaHatc.shape)
            print("\nBatac:", self.Batac.shape)
            print("Vc:", self.Vc.shape)
        return

    def predict(self, x):
        return sigmoid(np.dot(np.transpose(self.Batac[1] - self.Batac[0]), x) + (self.Vc[1] - self.Vc[0]))

    def sharedCovMatrix(self):
        return np.array(self.Nc[0] * self.SigmaHatc[0] + self.Nc[1] * self.SigmaHatc[1]) / (self.Nc.sum() - self.Nc.size)

# Import data
sales = pd.read_csv("data/vgsales.csv").drop(labels=["Name", "Year", "Platform", "Genre", "Global_Sales"], axis = 1)
sales['Publisher'].replace('', np.nan, inplace=True)
sales.dropna(subset=['Publisher'], inplace=True)

# Pull out two calsses and append the two
# Nintendo                         703
# Activision                       975
Class2 = sales[sales.Publisher == 'Nintendo']
Class1 = sales[sales.Publisher == 'Activision']
Xfull = Class1.append(Class2)

y = LabelEncoder().fit_transform(Xfull.Publisher.values)
X = Xfull.drop(labels=["Publisher"], axis = 1).values
#X = sklearn.preprocessing.StandardScaler().fit(X).transform(X)

min_max_scaler = MinMaxScaler()
X = min_max_scaler.fit_transform(X)

averageAccuracy = 0
folds = 10
for i, fold in enumerate(createCVData(X, folds)):
    XValid, yValid = X[fold], y[fold]
    XTrain = X[[index for index in range(X.shape[0]) if not index in fold]]
    yTrain = y[[index for index in range(y.shape[0]) if not index in fold]]

    lda = LDA(0)
    lda.fitXy(XTrain, yTrain)
    #lda.fitXyShared(XTrain, yTrain)
    predictions = np.array([lda.predict(test) for test in XValid]) >= 0.5
    acc = round((predictions == yValid).astype('uint8').sum()/yValid.size,2)
    print("Fold", i, ":", acc)
    averageAccuracy += acc

print("\nAverage:", round(averageAccuracy/folds,2))
