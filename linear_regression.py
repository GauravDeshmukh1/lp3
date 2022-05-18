#import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Read dataset
dataset = pd.read_csv("linear_regression_dataset.csv")
print(dataset.shape)
dataset.head()

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

#import linear regression and create object of it
from sklearn.linear_model import LinearRegression

#create object of LinearRegression
regressor = LinearRegression()

#train the data
regressor.fit(x,y) 

#print value of coefficient
print("Coefficient : ",regressor.coef_)

#print value of intercept
print("intercept : ",regressor.intercept_)

#calculate accuracy
accuracy = regressor.score(x,y)*100
print("Accuracy : ",accuracy)

y_pred = regressor.predict([[8]])
print(y_pred)

#take input from user and predict the value
hours = int(input("Enter the no of hours : "))
predicted_value = regressor.predict([[hours]])
print(predicted_value)

#plotting points 
plt.plot(x,y,'o', label="data point")

#plot the line
plt.plot(x,regressor.predict(x), color='#ff0000', label='regression line')

# x-axis label
plt.xlabel('Driving Hours')

#y-axis label
plt.ylabel('Risk Score')

plt.legend()
plt.show()
