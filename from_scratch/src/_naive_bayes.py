import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import math
from sklearn.metrics import confusion_matrix

data = pd.read_csv(r"../../../rust/_garage/data_banknote_authentication.txt")
X = np.array(data[data.columns[:-1]])
y = np.array(data[data.columns[-1]])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
gnb = GaussianNB()
gnb.fit(X_train,y_train)
print("Accuracy ",gnb.score(X_test,y_test))
print("Confusion matrix:\n",confusion_matrix(gnb.predict(X_test),y_test))

'''
PYTHON OUTPUT

Accuracy  0.8545454545454545
Confusion matrix:
 [[132  26]
 [ 14 103]]
'''