#importing libraries 
import numpy as np  
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv("Seattle_WeatherPredicton.csv",sep=",")
temp_max=df["temp_max"]
precipitation=df["precipitation"]


df.head(30)

x = np.array(temp_max).reshape(-1, 1)
y = np.array(precipitation)

#Splitting the data into Train and Test
from sklearn.model_selection import train_test_split 
xtrain, xtest, ytrain, ytest = train_test_split(  x, y, test_size=1/2, random_state=1 )

from sklearn.linear_model import LinearRegression 
regressor = LinearRegression()
regressor.fit( xtrain, ytrain )

print(df.shape)  #checking dimensions of Dataset

#Data Preprocessing to clean, organize the data  and read-to-feed to the Machine Learning model.

print(df.info()) #converting raw data to a suitable format

df.isnull().sum()*100/len(df)  #checking for null values

regressor.coef_ , regressor.intercept_   #y = mx + c  where m is coefficient , c is intercept

actualValue = ytrain
predictedValue = regressor.predict(xtrain) 
xtrain[0], actualValue[0] , predictedValue[0]

regressor.coef_ * xtrain[0] + regressor.intercept_ #y = mx + c

# Actual values
plt.scatter(xtrain, ytrain, color='blue') # x = xtrain , y = ytrain

#Predicted values
prediction = regressor.predict(xtrain)
plt.plot(xtrain, prediction , color = 'green') # y = prediction

plt.title ("Prediction for Training Dataset")
plt.xlabel("temp_max in degrees"), plt.ylabel("Precipitation")
plt.show()

plt.scatter(xtest, ytest, color= 'blue')

plt.plot(xtrain, regressor.predict(xtrain), color = 'orange')

plt.title ("Training Dataset:")
plt.xlabel("max_temp in degree"), plt.ylabel("Precipitation")
plt.show()

cor = df.corr()
plt.figure(figsize=(10,6))
sns.heatmap(cor,annot=True , cmap='coolwarm')
plt.show()

data = ["precipitation","temp_max","temp_min"]
for col in data:
    plt.figure(figsize=(20,10))
    plt.hist(df[col])
    plt.title(col)
    plt.show()

