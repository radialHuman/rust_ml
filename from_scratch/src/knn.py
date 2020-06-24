import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import math
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

data = pd.read_csv(r"../../../rust/_garage/data_banknote_authentication.txt")
X = np.array(data[data.columns[:-1]])
y = np.array(data[data.columns[-1]])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
decision_tree = KNeighborsClassifier()
decision_tree = decision_tree.fit(X_train,y_train)
y_predict = decision_tree.predict(X_test)

print("Accuracy : ", accuracy_score(y_test, y_predict))
print("Confusion matrix : \n",confusion_matrix(y_test, y_predict))

'''
PYTHON OUTPUT

Accuracy :  0.9963636363636363
Confusion matrix : 
[[166   1]
 [  0 108]]
'''