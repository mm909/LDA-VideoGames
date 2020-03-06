import math
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def predict(x, Batac, Vc):
    return sigmoid(np.dot(np.transpose(Batac[1] - Batac[0]), x) + (Vc[1] - Vc[0]))

# Import data
sales = pd.read_csv("data/vgsales.csv").drop(labels=["Name", "Year", "Platform", "Genre", "Global_Sales"], axis = 1)
# sales['Tenant'].replace('', np.nan, inplace=True)
# sales.dropna(subset=['Tenant'], inplace=True)
# Pull out two calsses and append the two
# Ubisoft                          921
# Nintendo                         703
Class1 = sales[sales.Publisher == 'Nintendo']
Class2 = sales[sales.Publisher == 'Ubisoft']
Xfull = Class1.append(Class2)

n = 1624 # n = 921 + 703
d = 5 # d = (Rank, NA, EU, JP, Other)

# Get and encode publisher
y = Xfull.Publisher
Nc = y.value_counts().values
y = LabelEncoder().fit_transform(y.values)
print(y)
PIc = Nc / n

X = Xfull.drop(labels=["Publisher"], axis = 1).values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10)

min_max_scaler = MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)
X_train = pd.DataFrame(X_train)
# X_test = min_max_scaler.transform(X_test)
# X_test = pd.DataFrame(X_test)

print('')
print('X.shape:', X.shape)
print('y.shape:', y.shape)
print('')
print('n   :', n)
print('Nc  :', Nc)
print('PIc :', PIc)
print('')

Uc = np.array([np.mean(X_train[y_train == 0], axis=0), np.mean(X_train[y_train == 1], axis=0)])
print("Uc       :", Uc.shape)

# Calc SigmaHatc
SigmaS1 = X_train[y_train == 0] - Uc[0]
SigmaSub1 = np.dot(SigmaS1.transpose(), SigmaS1) / Nc[0]
SigmaS2 = X_train[y_train == 1] - Uc[1]
SigmaSub2 = np.dot(SigmaS2.transpose(), SigmaS2) / Nc[1]
SigmaHatc = np.array([SigmaSub1, SigmaSub2])
print("SigmaHatc:", SigmaHatc.shape)

Batac = np.array([np.dot(np.linalg.inv(SigmaHatc[0]), Uc[0]), np.dot(np.linalg.inv(SigmaHatc[1]), Uc[1])])
print("\nBatac:", Batac.shape)

Vc = np.array([(-1/2) * np.dot(Uc[0].transpose(), np.dot(np.linalg.inv(SigmaHatc[0]), Uc[0])) + np.log(PIc[0]),
               (-1/2) * np.dot(Uc[1].transpose(), np.dot(np.linalg.inv(SigmaHatc[1]), Uc[1])) + np.log(PIc[1])])

print("Vc:", Vc.shape)


# testDataN = [1, 41.49, 29.02, 3.77, 8.46]
predictions = np.array([predict(test, Batac, Vc) for test in X_test]) >= 0.5

print("")
print(round((predictions == y_test).astype('uint8').sum()/y_test.size,2))
