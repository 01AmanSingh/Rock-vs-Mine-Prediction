https://colab.research.google.com/drive/1Ui0ZmXkOAuktOGHHRi-19xoNamTiaBBl?usp=sharing

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#loading the dataset to a pandas Dataframe
sonar_data = pd.read_csv('/content/sonar data.csv' , header = None)
sonar_data.head()
# number of rown and columns
sonar_data.shape
 sonar_data.describe()  #describe --> stastistical measures of the data
 sonar_data[60].value_counts()
 sonar_data.groupby(60).mean()
 # separating data and labels
X = sonar_data.drop(columns = 60 , axis = 1)
Y = sonar_data[60]
print(X)
print(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, stratify=Y, random_state=1)
print(X.shape, X_train.shape, X_test.shape)
print(X_train)
print(Y_train)
model = LogisticRegression()
#training logistics regression model with training data
model.fit(X_train, Y_train)
 X_train_prediction = model.predict(X_train)
 training_data_accuracy = accuracy_score(X_train_prediction, Y_train) 
 print('Accuracy on training data : ', training_data_accuracy) X_test_prediction = model.predict(X_test)
 test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
 print('Accuracy on test data : ', test_data_accuracy)
 input_data = ('')

#changing the input_data to numpy array
input_data_as_numpy_array = np.asarray(input_data)

#reshape the np array as we are predicting for one instance
imput_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(imput_data_reshaped)
print(prediction)

if (prediction[0] == 'R'):
  print('The object is a Rock')
else:
  print('The object is a Mine')




 
 print('Accuracy on training data : ', training_data_accuracy)
