# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import necessary libraries
2. Load the dataset and split into training and testing sets
3. Create and train the Linear Regression model
4.  Use the trained model to predict marks in the test dataset

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: MOHAMED HAMEEM SAJITH J
RegisterNumber:  212223240090
*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('/content/ml-lab-1.csv')
df.head(10)

plt.scatter(df['x'],df['y'])
plt.xlabel('x')
plt.ylabel('y')

x=df.iloc[:,0:1]
y=df.iloc[:,-1]
x

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression

lr=LinearRegression()
lr.fit(x_train,y_train)

x_train
y_train

lr.predict(x_test.iloc[0].values.reshape(1,1))

plt.scatter(df['x'],df['y'])
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x_train,lr.predict(x_train),color='red')
print("coefficient",lr.coef_)
print("intercept",lr.intercept_)

```

## Output:
# 1. HEAD:

   ![image](https://github.com/Sajith7862/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145972360/aaed63ef-4488-4a36-8e4e-7aaf912fcbaa)

# 2. GRAPH OF PLOTTED DATA:

   ![image](https://github.com/Sajith7862/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145972360/bb0ca266-592e-42ed-9e62-49b253ef2a01)

# 3.TRAINED DATA:

![image](https://github.com/Sajith7862/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145972360/2bb94dcb-d869-4c58-b84d-a3b74ec040f4)

# 4.LINE OF REGRESSION:

![image](https://github.com/Sajith7862/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145972360/e3d46f47-f7e7-4bd0-b872-1df58962c5e4)

# 5.COEFFICIENT AND INTERCEPT VALUES:

![image](https://github.com/Sajith7862/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/145972360/780a0b1e-4c46-44c0-ad16-c791eeec6429)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
