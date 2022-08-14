import pandas as pd
import numpy as np
import math
from operator import itemgetter
from random import shuffle

k = 5
n = 4
p = 0.5


AML=pd.read_csv('AML.dat', sep='\t')
amlData = AML.as_matrix()
amldata = np.insert(amlData, amlData.shape[0], 0, axis=0)

ALL = pd.read_csv('ALL.dat', sep = '\t')
allData = ALL.as_matrix()
alldata = np.insert(allData, allData.shape[0], 1, axis=0)

numAML = amldata.shape[1]
numTrainAML = int((n-1)/float(n) * numAML)
numALL = alldata.shape[1]
numTrainALL = int((n-1)/float(n) * numALL)
numTestAML = numAML - numTrainAML
numTestALL = numALL - numTrainALL

# This calculates the euclidean distance between two points
# point1 = first vector of features (i.e. aml[:, 1]
# point2 = second vector of features (should be the test set - we get # of dim from this)
# numDim = number of features, i.e. len(aml[:,1])
def eucDist(point1, point2):
	distance = 0
	numDim = len(point2) - 1
	for x in range(numDim):
	   distance += pow((point1[x] - point2[x]), 2)
	return math.sqrt(distance)

# This function returns the top K sorted neighbors in a list of lists
def nearestNeighbors(trainingSet, testInstance, k):
	neighbors = []
	dist = []
	length = len(testInstance) - 1 # this is the # of genes i.e. rows
	# loop through the training set, col by col, determine dist
	for i in range(trainingSet.shape[1]): # this is the # of columns in the training set
		dist = eucDist(trainingSet[:length,i], testInstance[:length,])
		#print("dist = %s")%dist
		neighbors.append([trainingSet[:,i], dist])
	sortedNeighbors = sorted(neighbors, key=itemgetter(1))
	return sortedNeighbors[:k]

# this function predicts the class from a list of the top K neighbors
def predictClass(topK):
        length = topK[0][0].shape[0]-1
        #print("length = %s")%length
    	votes = []
    	votesNeeded = float(len(topK))*p
	for i in range(len(topK)):
	    result = topK[i][0][length]
	    votes.append(result)
	#print("votes = %s")%votes
	if votes.count(1) >= votesNeeded:
	    return 1
	else:
	    return 0
    
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
        print("n = %s")%n
        amlTest = np.asarray(newAML[:, i::n]) # takes every k-th column
        #print("Cols to test: %s")%colTest
        colList = range(0, newAML.shape[1]) # Just a vector with all col nums of AML data for testing
        colsTrain = (filter(lambda x: ((x-i) % n != 0), colList))
        amlTrain = newAML[: , colsTrain]
        print("AML cols to train on: %s")%colsTrain
        
        colTest = colList[i::n]
        print("Cols to test: %s")%colTest
        
        allTest = np.asarray(newALL[:, i::n]) # takes every k-th column
        colList = range(0, newALL.shape[1]) # all colnums of ALL  data
        colsTrain = (filter(lambda x: ((x-i) % n != 0), colList))
        allTrain = newALL[: , colsTrain]
        print("ALLols to train on: %s")%colsTrain
            
        traindata = np.concatenate((amlTrain, allTrain), axis = 1)
        testdata = np.concatenate((amlTest, allTest), axis = 1)
        
        for z in range(testdata.shape[1]): #loop through cols of testdata
            testInstance = testdata[:,z]
            neighbors = nearestNeighbors(traindata, testInstance, k)
            pred = predictClass(neighbors)
            #print("Prediction = %s")%pred
            realClass = testdata[testdata.shape[0]-1, z]
            #print("realClass = %s")%realClass
            
            ## Now assess accuracy" AML = 0, ALL = 1
            if(pred == 0 and realClass == 0):
                TN += 1
            elif(pred == 0 and realClass == 1):
                FN += 1
            elif(pred == 1 and realClass == 1):
                TP += 1
            elif(pred == 1 and realClass == 0):
                FP += 1
    print("TP = %s")%TP
    print("TN = %s")%TN
    
    return TP, TN, FP, FN
  
#ts1 = amldata[:, :5]
#ts2 = alldata[:, :5]
#trainingSet = np.concatenate((ts1,ts2),axis=1)
#testInstance = amldata[:amldata.shape[0]-1, 6]
#neighbors = nearestNeighbors(trainingSet, testInstance, 3)

#classPred = predictClass(neighbors)
results = k_fold_CV(amldata, alldata, n)

sensitivity = float(results[0])/(results[0]+results[3])
specificity = float(results[1])/(results[1]+results[2])
accuracy = float(results[0]+results[1])/(results[0]+results[1]+results[2]+results[3])
print("Accuracy = %s")%accuracy