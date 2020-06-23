import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import math
from sklearn.metrics import confusion_matrix

data = pd.read_csv(r"../../../rust/_garage/data_banknote_authentication.txt")
X = np.array(data[data.columns[:-1]])
y = np.array(data[data.columns[-1]])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
reg = linear_model.LogisticRegression()
reg.fit(X_train,y_train)
print("Accuracy ",reg.score(X_test,y_test))
print("Weights ",reg.coef_)
print("Confusion matrix:\n",confusion_matrix(reg.predict(X_test),y_test))

'''
PYTHON OUTPUT

Accuracy  0.9963636363636363
Weights  [[-3.12754599 -1.74049223 -2.134707   -0.06385614]]
Confusion matrix:
 [[161   0]
 [  1 113]]
'''