import pandas as pd
import numpy as np
import math
from operator import itemgetter
import sys

''' 
Initialize global variables
    k = number of neighbors to consider
    n = folds of cross-validation
    p = percent of neighbors that must be "positive" for an observation to be classified as positive
    posFile = the filename of the file containing positive observations
    negFile = the filename of the file containing negative observations
'''
k = int(sys.argv[3])
n = int(sys.argv[5])
p = float(sys.argv[4])
posFile = sys.argv[1]
negFile = sys.argv[2]

'''
This function reads in the input data
Inputs:
    posFile = the filename of the file containing positive observations
    negFile = the filename of the file containing negative observations
Returns: 
    amldata = a numpy array containing all negative observations
    alldata = a numpy array containing all positive observations
'''
def readFiles(posFile, negFile):
    AML=pd.read_csv(negFile, sep='\t')
    amlData = AML.as_matrix()
    amldata = np.insert(amlData, amlData.shape[0], 0, axis=0)

    ALL = pd.read_csv(posFile, sep = '\t')
    allData = ALL.as_matrix()
    alldata = np.insert(allData, allData.shape[0], 1, axis=0)
    return amldata, alldata

'''
This function calculates the euclidean distance between two points
Inputs:
    point1 = first observation (a vector of features, n-dimensional point where n is the number of observations we have about each data point)
    point2 = second observations (second vector of features, n-dimensional point)
Returns the euclidean distance between these two points
'''
def eucDist(point1, point2):
    distance = 0
    numDim = len(point2) - 1
    for x in range(numDim):
        distance += pow((point1[x] - point2[x]), 2)
    return math.sqrt(distance)

'''
This function determines the k nearest neighbors to a test observation
Inputs:
    trainingSet: the set of training observations we are choosing the nearest neighbors from
    testInstance: the observation we are trying to determine the k nearest neighbors of (in the test set)
    k: the number of neighbors we are considering
Returns the top k nearest neighbors in a list of lists (each neighbor is a list)
'''
def nearestNeighbors(trainingSet, testInstance, k):
    neighbors = []
    dist = []
    length = len(testInstance) - 1 # this is the # of genes i.e. rows
   	# loop through the training set, col by col, determine dist
    for i in range(trainingSet.shape[1]): # this is the # of columns in the training set
        dist = eucDist(trainingSet[:length,i], testInstance[:length,])
        neighbors.append([trainingSet[:,i], dist])
    sortedNeighbors = sorted(neighbors, key=itemgetter(1))
    return sortedNeighbors[:k]

'''
This function predicts the class (positive vs. negative, 0 vs. 1) from a list of the top K neighbors
Inputs:
    topK: a list of lists, representing the k nearest neighbors to a given data point
Returns the predicted class (0 for negative, 1 for positive)of the test observation we are considering, 
based on whether there are at least k*p votes for the positive class (where p is the user-specified threshold, see above)
'''
def predictClass(topK):
    length = topK[0][0].shape[0]-1
    votes = list()
    votesNeeded = float(len(topK))*p
    for i in range(len(topK)):
        result = topK[i][0][length]
        votes.append(result)
    if votes.count(1) >= votesNeeded:
        return 1
    else:
        return 0
'''
This function performs n-fold cross validation of a k-nearest neighbors classifier to estimate the 
sensitivity, specificity, and accuracy of the classifier. 
Inputs:
    amldata: the array containing all negative observations/instances
    alldata: the array containing all positive observations/instances
    n: the number of folds specified by the user
Returns: TP, TN, FP, FN
    TP: True positives, the number of positive instances correctly classified as positive (1) by the classifier when they were in the test fraction
    TN: True negatives, the number of negative instances correctly classified as negative (0) by the classifier when they were in the test fraction
    FP: False positives, the number of negative instances incorrectly classified as positives by the classifier
    FN: False negatives, the number of positive instances incorrectly classified negatives by the classifier
'''
def k_fold_CV(amldata, alldata, n):
    TP, TN, FP, FN = 0, 0, 0, 0
    # Shuffle all the data
    transpAML = np.transpose(amldata) 
    np.random.shuffle(transpAML)
    newAML = np.transpose(transpAML).copy()
    transpALL = np.transpose(alldata)
    np.random.shuffle(transpALL)
    newALL = np.transpose(transpALL).copy()
    # Now loop through folds (0 through k)
    for i in range(0,n): # i denotes the "chunk" of CV we're on
        #print("n = %s")%n
        amlTest = np.asarray(newAML[:, i::n]) # takes every k-th column
        #print("Cols to test: %s")%colTest
        colList = range(0, newAML.shape[1]) # Just a vector with all col nums of AML data for testing
        colsTrain = (filter(lambda x: ((x-i) % n != 0), colList))
        amlTrain = newAML[: , np.asarray(colsTrain)]        
        allTest = np.asarray(newALL[:, i::n]) # takes every k-th column
        colList = range(0, newALL.shape[1]) # all colnums of ALL  data
        colsTrain = (filter(lambda x: ((x-i) % n != 0), colList))
        allTrain = newALL[: , colsTrain]
        traindata = np.concatenate((amlTrain, allTrain), axis = 1)
        testdata = np.concatenate((amlTest, allTest), axis = 1)
        
        for z in range(testdata.shape[1]): #loop through cols of testdata
            testInstance = testdata[:,z]
            neighbors = nearestNeighbors(traindata, testInstance, k)
            pred = predictClass(neighbors)
            realClass = testdata[testdata.shape[0]-1, z]            
            if(pred == 0 and realClass == 0):
                TN += 1
            elif(pred == 0 and realClass == 1): # This is ALL classified by AML
                FN += 1
            elif(pred == 1 and realClass == 1):
                TP += 1
            elif(pred == 1 and realClass == 0):
                FP += 1
    return TP, TN, FP, FN
'''
This function writes summary info about the cross-validated performance of the classifier to the output file "knn.out"
Inputs:
    results: a tuple of floats, representing the true positives, true negatives, false positives, and false negatives
        of the classifier on the training data (in the order TP, TN, FP, FN)
knn.out file outputs:
    k = number of neighbors considered by the classifier
    p = fraction of positive neighbors necessary for a "positive" classification of the test instance
    n = folds of cross-validation
    accuracy = the fraction of total correctly-classified instances
    sensitivity = true negatives / all negative instances, aka TN / (TN + FP)
    specificity = true positives / all positive instances, aka TP / (TP + FN)
'''
def printOutput(results):
    sensitivity = float(results[0])/(results[0]+results[3])
    specificity = float(results[1])/(results[1]+results[2])
    accuracy = float(results[0]+results[1])/(results[0]+results[1]+results[2]+results[3])
    outfile = open('knn.out', 'w')
    outfile.write("k: ")
    outfile.write(str(k))
    outfile.write("\np: ")
    outfile.write(str("{0:.2f}".format(p)))
    outfile.write("\nn: ")
    outfile.write(str(n))
    outfile.write("\naccuracy: ")
    outfile.write(str("{0:.2f}".format(accuracy)))
    outfile.write("\nsensitivity: ")
    outfile.write(str("{0:.2f}".format(sensitivity)))
    outfile.write("\nspecificity: ")
    if specificity >= 0.995:
        outfile.write(str(1.00))
    else:
        outfile.write(str("{0:.2f}".format(specificity)))

'''
This code calls the main functions the program to read the data, compute k-fold CV of KNN, and print results
'''
amldata, alldata = readFiles(posFile, negFile)
results = k_fold_CV(amldata, alldata, n)
printOutput(results)