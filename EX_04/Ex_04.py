
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn import preprocessing

XX=[]

df=pd.read_csv('roasting_data.csv')
X=np.array(df[[
    'T_data_1_1','T_data_1_2','T_data_1_3'
    ,'T_data_2_1','T_data_2_2','T_data_2_3'
    ,'T_data_3_1','T_data_3_2','T_data_3_3'
    ,'T_data_4_1','T_data_4_2','T_data_4_3'
    ,'T_data_5_1','T_data_5_2','T_data_5_3'
    ,'H_data'
    ,'AH_data']])



Y=np.array(df[['quality']])
Xtrain, Xtest,Ytrain, Ytest=train_test_split(X, Y,test_size=0.3)
model=linear_model.LinearRegression()
model.fit(Xtrain, Ytrain)
trainprediction = model.predict(Xtrain)
testprediction = model.predict(Xtest)

print('Mean absolute error in train data %.2f'% mean_absolute_error(Ytrain,trainprediction))
print('Mean absolute error in test data %.2f'% mean_absolute_error(Ytest,testprediction))
print('R2 score in train data %.2f'% r2_score(Ytrain,trainprediction))
print('R2 score in test data %.2f'% r2_score(Ytest,testprediction))
#%%
M_AB_Error=[]
string_array = np.array(['T_data_1_1','T_data_1_2','T_data_1_3'
                        ,'T_data_2_1','T_data_2_2','T_data_2_3'
                        ,'T_data_3_1','T_data_3_2','T_data_3_3'
                        ,'T_data_4_1','T_data_4_2','T_data_4_3'
                        ,'T_data_5_1','T_data_5_2','T_data_5_3'])
for col in range(15):
    new_arr = np.delete(X, col, axis=1)
    Xtrain, Xtest,Ytrain, Ytest=train_test_split(new_arr, Y,test_size=0.3)
    model=linear_model.LinearRegression()
    model.fit(Xtrain, Ytrain)
    trainprediction = model.predict(Xtrain)
    testprediction = model.predict(Xtest)
    M_AB_Error.append(mean_absolute_error(Ytrain,trainprediction))
    
index_of_sensor=np.argmax(M_AB_Error)
print("The name of the sensor that has the most impact:",string_array[index_of_sensor])
print("The index of the sensor that has the most impact:",index_of_sensor)
