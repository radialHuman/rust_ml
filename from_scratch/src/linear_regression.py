import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
import math
from sklearn.metrics import mean_squared_error, r2_score


data = pd.read_csv(r"../../../rust/_garage/ccpp.csv")
X = np.array(data[data.columns[:-1]])
y = np.array(data[data.columns[-1]])
for i in [20,25,30,35]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=i/100)
    reg = linear_model.LinearRegression()
    reg.fit(X,y)
    print("The coeficients of a columns as per simple linear regression on {} of data is :{}".format(i ,reg.coef_))
    print("MSE:", mean_squared_error(y_test,reg.predict(X_test)))
    print("RMSE:", math.sqrt(mean_squared_error(y_test,reg.predict(X_test))))
    print("MAPE: ", np.mean(np.abs((y_test - reg.predict(X_test)) / y_test)) * 100)
    print("R2:", r2_score(y_test,reg.predict(X_test)))
    print()

'''
PYTHON OUTPUT

The coeficients of a columns as per simple linear regression on 20 of data is :[-1.97751311 -0.23391642  0.06208294 -0.1580541 ]
MSE: 20.32289802862265
RMSE: 4.50809250444383
MAPE:  0.7944134634162926
R2: 0.9297223453952952

The coeficients of a columns as per simple linear regression on 25 of data is :[-1.97751311 -0.23391642  0.06208294 -0.1580541 ]
MSE: 20.349259622952545
RMSE: 4.511015364965248
MAPE:  0.7920410958466909
R2: 0.9321217668190203

The coeficients of a columns as per simple linear regression on 30 of data is :[-1.97751311 -0.23391642  0.06208294 -0.1580541 ]
MSE: 22.168361253669495
RMSE: 4.708328923691451
MAPE:  0.8186758130698937
R2: 0.9243771113857681

The coeficients of a columns as per simple linear regression on 35 of data is :[-1.97751311 -0.23391642  0.06208294 -0.1580541 ]
MSE: 20.359808440477174
RMSE: 4.512184442205037
MAPE:  0.7922430740280904
R2: 0.928566836595381

'''