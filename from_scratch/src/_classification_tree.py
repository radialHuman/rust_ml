import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import math
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

data = pd.read_csv(r"../../../rust/_garage/data_banknote_authentication.txt")
X = np.array(data[data.columns[:-1]])
y = np.array(data[data.columns[-1]])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
decision_tree = DecisionTreeClassifier(random_state=0, max_depth=2)
decision_tree = decision_tree.fit(X_train,y_train)
y_predict = decision_tree.predict(X_test)

print("Accuracy : ", accuracy_score(y_test, y_predict))
print("Confusion matrix : ",confusion_matrix(y_test, y_predict))


'''
PYTHON OUTPUT

0.8981818181818182
[[142  14]
 [ 14 105]]

'''