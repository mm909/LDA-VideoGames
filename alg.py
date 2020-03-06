import math
import pandas as pd
import numpy as np

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def predict(x, Batac, Vc):
    return sigmoid(np.dot(np.transpose(Batac[1] - Batac[0]), x) + (Vc[1] - Vc[0]))

n = 6
d = 2

dummy = pd.read_csv("data/dummy.csv")
print(dummy.values)

y = dummy.Answer.values
X = dummy.drop(labels=["Answer"], axis = 1).values
Nc = np.array([3,3])
PIc = Nc / n

print('')
print('X.shape:', X.shape)
print('y.shape:', y.shape)
print('')
print('n   :', n)
print('Nc  :', Nc)
print('PIc :', PIc)
print('')

Uc = np.array([np.mean(X[y == 0], axis=0), np.mean(X[y == 1], axis=0)])
print("Uc.shape :", Uc.shape)
print(Uc)

SigmaS1 = X[y == 0] - Uc[0]
SigmaSub1 = np.dot(SigmaS1.transpose(), SigmaS1) / Nc[0]
SigmaS2 = X[y == 1] - Uc[1]
SigmaSub2 = np.dot(SigmaS2.transpose(), SigmaS2) / Nc[1]
SigmaHatc = np.array([SigmaSub1, SigmaSub2])
print("SigmaHatc:", SigmaHatc.shape)

Batac = np.array([np.dot(np.linalg.inv(SigmaHatc[0]), Uc[0]), np.dot(np.linalg.inv(SigmaHatc[1]), Uc[1])])
print("\nBatac:", Batac.shape)

Vc = np.array([(-1/2) * np.dot(Uc[0].transpose(), np.dot(np.linalg.inv(SigmaHatc[0]), Uc[0])) + np.log(PIc[0]),
               (-1/2) * np.dot(Uc[1].transpose(), np.dot(np.linalg.inv(SigmaHatc[1]), Uc[1])) + np.log(PIc[1])])

print("Vc:", Vc.shape)

# test = [4.5, 7]
test = [8, 10]
predictions = np.array(predict(test, Batac, Vc)) >= 0.5
print(predictions)
#
# print("")
# print(round((predictions == y_test).astype('uint8').sum()/y_test.size,2))
