import chemoUtils as util
import sys
import numpy as np

'''
This function reads the nodes file and returns the protein nodes as an array
parameters: nodesFile = the file path to the file specifying all protein nodes we are interested in
Returns: nodes, a np array
    Column 1: uniprot accession
    Column 2: uniprot id
    Colun 3: indications
'''
def readNodes(nodesFile):
    nodes = np.loadtxt(nodesFile, dtype='string', delimiter=',', skiprows=1)
    return nodes

''' 
This function takes in two proteins and returns a list of the two proteins in alphabetical order
Parameters: protein1 and protein2: the names of two proteins to be sorted
returns: proteinList: a list of the two protein names (strigs) sorted by alphabetical order
'''
def sortProteins(protein1, protein2):
    proteinList = []
    proteinList.append(protein1)
    proteinList.append(protein2)
    proteinList.sort()
    return proteinList

'''
This program calculates all the nodes and edges in the protein network by calculating all bootstrapped p-values < 0.05.
It writes the necessary output to the network.sif and two *.nodeAttr files

Arguments:
    First argument: the filename of the drug file
    Second argument: filename of the protein file
    Third argument: filename of the target file
Output: returns nothing, but writes to three output files
    network.sif: a file containing all proteins (nodes) with bootstrapped p-values < 0.05 between it and another protein (these are the "edges), sorted alphabetically
    name.nodeAttr: contains labels of nodes with the SWISSPROT name from protein_nodes.csv
    indication.nodeAttr: labels the protein with a semi-colon separated list of all the indications associated with a given protein
'''
def main(): 
    drugFile = sys.argv[1]
    targetFile = sys.argv[2]
    nodesFile = sys.argv[3]
    targets = util.readTargetsData(targetFile)
    drugIDs, fingerprints = util.readDrugData(drugFile)
    nodes = readNodes(nodesFile)
    proteinList = nodes[:, 0]
    
    sifFile = open('network.sif', 'w')
    nameFile = open('name.nodeAttr', 'w'); 
    indicationFile = open('indication.nodeAttr', 'w'); 
    
    nodeList = []
    nameFile.write("name\n")
    indicationFile.write("indication\n")
    
    written = set()
    
    for i in range(0, len(proteinList)):
        for j in range(i + 1, len(proteinList)):
            protein1 = proteinList[i]
            ligands1 = util.getLigands(protein1, targets)
            protein2 = proteinList[j]
            ligands2 = util.getLigands(protein2, targets)
            numDrug1 = len(ligands1)
            numDrug2 = len(ligands2)
            real_score = util.tanimotoSummary(drugIDs, fingerprints, ligands1, ligands2)
            pval = util.calcBootstrap(100, real_score, numDrug1, numDrug2, drugIDs, fingerprints, 214)
            if pval <= 0.05:
                sortedProts = sortProteins(protein1, protein2)
                nodeList.append(sortedProts)
                if not (protein1 in written):
                    nameFile.write("%s = %s\n" % (protein1, nodes[i,1]))
                    indicationFile.write("%s = %s\n" % (protein1, nodes[i,2]))
                    written.add(protein1)
                if not (protein2 in written):
                    nameFile.write("%s = %s\n"%(protein2, nodes[j,1]))
                    indicationFile.write("%s = %s\n"%(protein2, nodes[j,2]))
                    written.add(protein2)
    nodeList.sort()
    for nodeItem in nodeList:
        sifFile.write("%s edge %s\n" % (nodeItem[0], nodeItem[1]))
    sifFile.close()
    nameFile.close()
    indicationFile.close()
            
            
if __name__ == "__main__":
    main()
    