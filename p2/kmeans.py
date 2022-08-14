import numpy as np
import sys
import pandas as pd
import math
from operator import itemgetter
import random as rand

copy = []
centroids = []
k = sys.argv[1]
inFile = sys.argv[2]
max_iterations = sys.argv[3]
iterations = 1
if(len(sys.argv) == 5):
    read_centroids = True
    centroidFile = sys.argv[4]
else:
    read_centroids = False
    
#sys.stdout.write(("Max iterations: %s \n")%str(max_iterations))
#sys.stdout.write(("read_centroids: %s \n")%str(read_centroids))

data = pd.read_csv(inFile, sep='\t', header = None)
inputdata = data.as_matrix()

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
def init_centroids(data):
    if read_centroids:
        data1 = pd.read_csv(centroidFile, sep='\t', header = None)
        centroidMatrix = data1.as_matrix()
        centroids = centroidMatrix.tolist()
        centroids = centroids[:int(k)]
        #print("Centroids = %s")%centroids
    else:
        copy = np.asarray(list(inputdata))
        np.random.shuffle(copy)
        centroids = copy[:int(k), :].tolist()
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
    #print(str(cluster_dict))
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
    outfile = open('kmeans.out', 'w')
    i = 1
    for point in range(inputdata.shape[0]):
        outfile.write(str(i))
        outfile.write("\t")
        key = tuple(inputdata[point].tolist())
        outfile.write(str(dictionary[key]))
        outfile.write("\n")
        i += 1

centroids = init_centroids(inputdata)
init_clusters = runCluster(inputdata, centroids)
new_centroids = newCentroids(init_clusters, centroids)
new_dict = dict()
while not hasConverged(new_centroids, centroids, iterations):
    new_dict = runCluster(inputdata, new_centroids)
    centroids = new_centroids[:]
    new_centroids = newCentroids(new_dict, centroids)[:]
    iterations += 1
if len(new_dict.keys()) == 0:
    print_output(init_clusters)
else:
    print_output(new_dict)
sys.stdout.write("iterations: ")
sys.stdout.write(str(iterations))
sys.stdout.write("\n")