import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn import model_selection 
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import accuracy_score 
from sklearn.linear_model import LogisticRegression 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis 
from sklearn.naive_bayes import GaussianNB 
from sklearn.svm import SVC 

dataset=pd.read_csv("auto-mpg.csv")
#print(dataset)
print(dataset.head())
#print(dataset.describe())
#print(dataset.groupby('mpg').size())

#dataset.hist()
#plt.show()

#dataset.plot(kind='density', subplots=True, layout=(3,3), sharex=False)
#plt.show()

#_,ax = plt.subplots () 
#ax.hist(dataset['acceleration'],bins=20,color='red',alpha=0.8,label="acceleration")
#ax.hist(dataset['mpg'],bins=20,color='red',alpha=0.5,label="mpg")
#ax.set_title('Acceleration vs mpg')
#ax.set_ylabel('Frequency')
#ax.set_xlabel('Acceleration vs mpg')
#ax.legend(loc='best')
#plt.show()

#dataset.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False)
#plt.show()

#sns.heatmap(dataset.corr())
#plt.show()

array=dataset.values
X=array[:,0:7]#all rows from 0 to 6
Y=array[:,7]# 6 counted here


# Splitting dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X, Y, test_size=0.2, random_state=0)

#Fitting multiple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
print('Learning completed')
# Prdeicting the test results
Y_prediction=regressor.predict(X_test)
print(Y_prediction)

#Calculating the coefficients
print(regressor.coef_)
print(regressor.intercept_)

# Validating how good our prediction(model) is
from sklearn.metrics import r2_score
print(r2_score(Y_test,Y_prediction))


#from sklearn.metrics import confusion_matrix
#print(confusion_matrix(Y_test,Y_prediction))
#newvalues=[[8,351,142,4054,14.3,79,1]]
#observation=regressor.predict(newvalues)
#print('predicted;',observation)

#print(classification_report(Y_test,Y_prediction))
from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors=3) 
knn.fit(X_train,Y_train)
print('Learning completed')
