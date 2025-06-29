
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt


df=pd.read_csv('HousePrice.csv')
X=np.array(df[['GrLivArea']])
Y=np.array(df[['SalePrice']])

Xtrain, Xtest,Ytrain, Ytest=train_test_split(X, Y,test_size=0.3)

model=linear_model.LinearRegression()
model.fit(Xtrain, Ytrain)

trainprediction = model.predict(Xtrain)
testprediction = model.predict(Xtest)


print('Mean absolute error in train data %.2f'% mean_absolute_error(Ytrain,trainprediction))
print('Mean absolute error in test data %.2f'% mean_absolute_error(Ytest,testprediction))

print('R2 score in train data %.2f'% r2_score(Ytrain,trainprediction))
print('R2 score in test data %.2f'% r2_score(Ytest,testprediction))

plt.figure()
plt.scatter(Xtrain, Ytrain, label='Actual prices')
plt.scatter(Xtrain, trainprediction, label='predicted prices')
plt.title('Train data prediction')
plt.xlabel('Living area')
plt.ylabel('Sale price')
plt.legend()

plt.figure()
plt.scatter(Xtest, Ytest, label='Actual prices')
plt.scatter(Xtest, testprediction, label='predicted prices')
plt.title('test data prediction')
plt.xlabel('Living area')
plt.ylabel('Sale price')
plt.legend()

