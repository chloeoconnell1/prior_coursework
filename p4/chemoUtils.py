import sys
import numpy as np
from numpy import random

'''
This function calculates all tanimoto scores between all drugs
Parameters: 
    drugsfile: the filename that lists all db_ids, generic_names, and features (maccs) of each drug
    targetsfile: the filename of the file specifying drug-target pairs (db_id, uniprot_accession, uniprot_id)
Returns: a numpy array of pairwise drug scores
    Column 1: drugbank ID #1
    Column 2: drugbank ID #2
    Column 3: Tanimoto score (6 decimal places) between the two
    Column 4: 1 or 0 indicating whether the drugs share a target or not (1 = target shared)
'''
def allScores(drugsfile, targetsfile):
    drugIDs, fingerprints = readDrugData(drugsfile)
    targets = readTargetsData(targetsfile)
    scores = []
    # Loop through all drug pairs and calculate t-scores, convert list to array
    for i in range(0, len(drugIDs)): 
        for j in range(i+1, len(drugIDs)): # Don't want to calculate symmetrical scores we've already calculated
            score = calcScore(i, j, fingerprints)
            drug1 = drugIDs[i]
            drug2 = drugIDs[j]
            ifShared = shared(drug1, drug2, targets)
            scoresList = [drug1, drug2, format(score,'.6f'), ifShared]
            scores.append(scoresList)
    scoreArray = np.asarray(scores)
    return scoreArray
'''
This function determines whether or not two drugs have a shared target
Parameters:
    drug1: the drugbank ID of the first drug
    drug2: the drugbank ID of the 2nd drug
    targets: an array representing information on drug targets
        column 1 = drugbank ID
        column 2 = target UniProt accession number 
        column 3 = target UniProt ID
'''
def shared(drug1, drug2, targets):
    index1 = np.where(targets == drug1)[0]
    targ1 = targets[index1, 1]
    index2 = np.where(targets == drug2)[0]
    targ2= targets[index2, 1]
    if len(set(targ1).intersection(set(targ2))) > 0:
        return 1
    else:
        return 0
    
'''
This function reads in drug information from the user-specified drugs file and returns info on drugs and their fingerprints
Parameters: 
    drugsfile: the filename of a csv specifying drug db_ids, generic_names, and features/fingerprints (maccs) of each drug
Returns:
    drugIDs: a list of drug IDs (strings)
    fingerprints: a list of lists representing the fingerprint values of each drug
'''
def readDrugData(drugsFile):
    drugIDs = []
    fingerprints = []
    bigArray = np.loadtxt(drugsFile, dtype='string', delimiter=',', skiprows=1)
    drugIDs = bigArray[:,0].tolist()
    fingerprintsList = bigArray[:,2].tolist() # yields a list of long strings
    fingerprints = []
    for item in fingerprintsList:
        fingerprints.extend([item.split()])
    return drugIDs, fingerprints


'''
This function reads the targets information from the targets input file
Parameters: targetsfile = the filename of the targets file, containing the following columns:
    drug db_id
    uniprot_accession
    uniprot_ID
Returns: an array with the same columns as above
'''
def readTargetsData(targetsfile):
    targets = np.loadtxt(targetsfile, dtype='string', delimiter=',', skiprows=1)
    return targets

'''
This function calculates the tanimoto similarity score between two drugs given their fingerprints
Parameters:
    drugIndex1: the index of the first drug in the fingerprints list
    drugIndex2: the index of the 2nd drug in the fingerprints list
    fingerprints: a list of lists representing the fingerprints of all drugs
returns: the Tanimoto Coefficient (also known as the Jaccard Index) of the drug pair
'''
def calcScore(drugIndex1, drugIndex2, fingerprints):
    set1 = set(fingerprints[drugIndex1])
    set2 = set(fingerprints[drugIndex2])
    intersect = len(set1.intersection(set2))
    return float(intersect)/float((len(set1) + len(set2) - intersect))

'''
This function returns all the ligands of a protein
Parameters: 
    protein: the uniprot accession number of the protein in question
    targets: the array representing drug target information (columns = drug db_id, uniprot accession, and uniprot ID, with one row for each drug target pair)
''' 
def getLigands(protein, targets):
    index = np.where(targets == protein)[0]
    ligands = list(targets[index,0])
    return ligands

'''
This function computs the tanimoto summary score between two proteins
Parameters:
    drugIDs: the list of drug IDs (strings)
    fingerprints: a list of lists, where each index of the larger list represents a different drug ID, and within that index, there is a list of all features of that drug
    ligandsA: a list of drugs that the first protein in question binds to
    ligandsB: a list of drugs that the 2nd protein in question binds to
Returns: a tanimoto summary score, representing the sum of all individual pairwise tanimoto scores between the protein ligands that are > 0.5 
'''
def tanimotoSummary(drugIDs, fingerprints, ligandsA, ligandsB):
    #print(drugIDs)
    summaryScore = float(0.0)
    for i in range(0, len(ligandsA)): # here must loop through ALL pairs of ligands
        for j in range(0, len(ligandsB)):
            if ligandsA[i] == ligandsB[j]:
                score = 1.0
            else:
                index1 = drugIDs.index(ligandsA[i])
                index2 = drugIDs.index(ligandsB[j])
                score = calcScore(index1, index2, fingerprints)
            if score > 0.5:
                summaryScore += score
    return summaryScore

'''
This function calculates the bootstrap p-value of the tanimoto summary score between two proteins
Parameters:
        n: number of bootstrap iterations desired
        real_score: the calculated tanimoto summary score between the two proteins
        numDrugsA: the number of drugs that bind to the first protein in question
        numDrugsB: the number of drugs that bind to the second protein in question
        drugIDs: the list of drug IDs 
        fingerprints: a list of lists, where each index corresponds to a drug and within that index, there is a list of drug features that make up that drug's fingerprint
        seed: the seed to be set for random number generation
Returns: a bootstrapped p-value for the tanimoto summary score between the two proteins
'''
def calcBootstrap(n, real_score, numDrugsA, numDrugsB, drugIDs, fingerprints, seed):
    random.seed(seed)
    counts = 0
    for i in range(0, n):
        rand1 = np.random.choice(drugIDs, size = numDrugsA)
        rand2 = np.random.choice(drugIDs, size = numDrugsB)
        bootstrapScore = tanimotoSummary(drugIDs, fingerprints, rand1, rand2)
        if bootstrapScore >= real_score:
            counts = counts + 1
    bootstrap_pval = float(counts) / n
    return bootstrap_pval