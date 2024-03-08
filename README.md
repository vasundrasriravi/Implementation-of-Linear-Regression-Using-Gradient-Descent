# Implementation-of-Linear-Regression-Using-Gradient-Descent

## AIM:
To write a program to predict the profit of a city using the linear regression model with gradient descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas, numpy and mathplotlib.pyplot.
2. Trace the best fit line and calculate the cost function.
3. Calculate the gradient descent and plot the graph for it.
4. Predict the profit for two population sizes.

## Program:
```
Program to implement the linear regression using gradient descent.
Developed by:VASUNDRA SRI R 
RegisterNumber:212222230168
```
```
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(X1,y,learning_rate=0.1,num_iters=1000):
    X=np.c_[np.ones(len(X1)),X1]
    theta=np.zeros(X.shape[1]).reshape(-1,1)
    
    for _ in range(num_iters):
        #Calculate predictions
        predictions=(X).dot(theta).reshape(-1,1)
        
        #Calculate errors
        errors=(predictions-y).reshape(-1,1)
        #Update theto using gradient descent
        theta=learning=learning_rate*(1/len(X1))*X.T.dot(errors)
    return theta

data=pd.read_csv("50_Startups.csv")
data.head()

X=(data.iloc[1:,:-2].values)
X1=X.astype(float)

scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X)
print(X1_Scaled)

#Learn model Parameters
theta= linear_regression(X1_Scaled,Y1_Scaled)
#Predict data value for a new value point
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
```
## Output:
## data.head():

![Screenshot 2024-03-08 113648](https://github.com/vasundrasriravi/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119393983/0df877a4-484e-425e-96e8-4e61927a45c9)

## x and x_Scaled:
![Screenshot 2024-03-08 114406](https://github.com/vasundrasriravi/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119393983/d504cd87-de98-460c-899d-1bf85a9f2a42)
![Screenshot 2024-03-08 114438](https://github.com/vasundrasriravi/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119393983/23b9a67a-7588-44f7-bcca-53de340e028b)

## Predicted values:
![Screenshot 2024-03-08 114143](https://github.com/vasundrasriravi/Implementation-of-Linear-Regression-Using-Gradient-Descent/assets/119393983/0eee4ecb-fc91-4211-b11b-1c7cc7c35406)


## Result:
Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
