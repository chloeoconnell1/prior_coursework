import os
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import numpy

os.chdir("/Users/titashde/Documents/BMI214/assignment2")

import pandas as pd
mydata=pd.read_csv('leukemia.csv', sep=',')

dataframe = mydata.as_matrix()

features = dataframe[:, 0:149]
result = dataframe[:, 150]


knn = KNeighborsClassifier(n_neighbors=5)

from sklearn.cross_validation import cross_val_score
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import LeaveOneOut
loo = LeaveOneOut()
loo.get_n_splits(features)

AMLtruePos = 0
AMLfalsePos = 0
numPred = 0 
ALLtruePos = 0
ALLfalsePos = 0
correct = 0
predList = []
for train_index, test_index in loo.split(features):
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = result[train_index], result[test_index]
    knn.fit(X_train, y_train)
    pred = knn.predict(X_test)
    predList.append(pred)
    if (pred == "AML") and (result[test_index]) == "AML":
        AMLtruePos += 1
        correct += 1
    elif (pred == "ALL") and (result[test_index]) == "AML":
        ALLfalsePos += 1
    elif(pred == "ALL") and (result[test_index]) == "ALL":
        ALLtruePos += 1
        correct += 1
    elif(pred == "AML") and (result[test_index]) == "ALL":
        AMLfalsePos += 1
    numPred += 1

accuracy = float(correct)/numPred
AMLtpr = float(AMLtruePos)/(AMLtruePos + ALLfalsePos) # true pos / all AML cases
AMLfpr = float(AMLfalsePos)/(AMLfalsePos + ALLtruePos) # negative instances will be all ALL cases
ALLtpr = float(ALLtruePos)/(ALLtruePos + AMLfalsePos)
ALLfpr = float(ALLfalsePos)/(AMLtruePos + ALLfalsePos) # false ALL pos / all AML cases (fp + tn)



from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

AMLtruePos = 0
AMLfalsePos = 0
numPred = 0 
ALLtruePos = 0
ALLfalsePos = 0
correct = 0
predList = []
for train_index, test_index in loo.split(features):
    X_train, X_test = features[train_index], features[test_index]
    y_train, y_test = result[train_index], result[test_index]
    gnb.fit(X_train, y_train)
    pred = gnb.predict(X_test)
    predList.append(pred)
    if (pred == "AML") and (result[test_index]) == "AML":
        AMLtruePos += 1
        correct += 1
    elif (pred == "ALL") and (result[test_index]) == "AML":
        ALLfalsePos += 1
    elif(pred == "ALL") and (result[test_index]) == "ALL":
        ALLtruePos += 1
        correct += 1
    elif(pred == "AML") and (result[test_index]) == "ALL":
        AMLfalsePos += 1
    numPred += 1

accuracy = float(correct)/numPred
AMLtpr = float(AMLtruePos)/(AMLtruePos + ALLfalsePos) # true pos / all AML cases
AMLfpr = float(AMLfalsePos)/(AMLfalsePos + ALLtruePos) # false pos / negative instances (will be all ALL cases)
ALLtpr = float(ALLtruePos)/(ALLtruePos + AMLfalsePos)
ALLfpr = float(ALLfalsePos)/(AMLtruePos + ALLfalsePos) # ALL false pos / all negative (all AML cases)

conf = confusion_matrix(result, predList)
