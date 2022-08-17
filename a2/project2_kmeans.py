/* Kmeans Algorithm: 
    This clusters data points so that each data point is assigned to a cluster.
    The objective of the kmeans algorithm is to minimize the average squared distance between each point and its cluster.
    The algorithm accepts three arguments:

    K: the number of clusters
    Input_file: the input file
    Max_Iterations: the maximum number of iterations
    (Optional) Centroid_file: the file containing the Starting centroids of the clusters. 
*/

import numpy as np
import sys
import pandas as pd
import math
from operator import itemgetter
import random as rd

centroids = []
k = sys.argv[1]
inFile = sys.argv[2]
max_iterations = sys.argv[3]
if(len(sys.argv) == 4):
    read_centroids = True
    centroidFile = sys.argv[4]
else:
    read_centroids = False

def hasConverged(centroids, old_centroids, iterations):
    #print("checking centroids for equality")
    #print("centroids = %s")%centroids
    #print("old_centroids = %s")%old_centroids
    if iterations > max_iterations:
        return True
    elif old_centroids == centroids:
        return True
    else:
        return False

#Issue is that centroids are sometimes getting made in the same place (i.e. same row)
def init_centroids(data, k):
    if read_centroids:
        data1 = pd.read_csv(centroidFile, sep='\t', header = None)
        centroidMatrix = data1.as_matrix()
        centroids = centroidMatrix.tolist()
        print("Centroids = %s")%centroids
    else:
        centroids = []
        random = rd.sample(range(0, data.shape[0]), k)
        #print("randomNums = %s")%random
        for i in random:
            new = data[i, :].flatten().tolist()
            centroids.append(new)
        print("centroids = %s")%centroids
    return centroids
        # centroids will be a list of lists
    
# Calculate the euclidean distance btw point and centraoids, return index of closest centroid
def closestCentroid(point, centroids):
    distances = []
    numDim = len(point)
    for i in range(len(centroids)):
        distance = 0
        for x in range(numDim):
            distance += pow((point[x] - centroids[i][x]), 2)
        distances.append(math.sqrt(distance))
    closestGroup = min(enumerate(distances), key=itemgetter(1))[0] 
    return closestGroup

def runCluster(inputdata, centroids):
    cluster_dict = dict()
    key = tuple()
    for i in range(0, len(inputdata)):
        group = closestCentroid(inputdata[i].tolist(), centroids)
        key = tuple(inputdata[i].tolist())
        cluster_dict[key] = group
        # Dict contains the point as a tuple key and the cluster as its value
    print(str(cluster_dict))
    return cluster_dict
    

def newCentroids(cluster_dict, centroids):
    newCentroids = []
    for i in range(0, len(centroids)):
        pointList = []
        for point, cluster in cluster_dict.items():
            if cluster == i:
                pointList.append(point)
        currCluster = np.asarray(pointList)
        newCentroids.append(computeCentroid(currCluster))
    return newCentroids
    

def computeCentroid(currCluster):
    length = currCluster.shape[0]
    numDim = currCluster.shape[1]
    centroid = []
    for i in range(0, numDim):
        total = float(np.sum(currCluster[:, i]))
        centroid.append(total / length)
    return centroid

def print_output(dictionary):
    outfile = open('knn.txt', 'w')
    i = 1
    for key in dictionary:
        outfile.write(str(i))
        outfile.write("\t")
        outfile.write(str(dictionary[key] + 1))
        outfile.write("\n")
        i += 1

data = pd.read_csv('testdata.dat', sep='\t', header = None)
inputdata = data.as_matrix()


centroids = init_centroids(inputdata, k)
init_clusters = runCluster(inputdata, centroids)
new_centroids = newCentroids(init_clusters, centroids)
while not hasConverged(new_centroids, centroids, iterations):
    new_dict = runCluster(inputdata, new_centroids)
    centroids = new_centroids[:]
    new_centroids = newCentroids(new_dict, centroids)[:]
    iterations += 1

print_output(new_dict)
sys.stdout.write(str(iterations))
