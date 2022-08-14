import argparse
import chemoUtils as util
import numpy as np
'''
This method adds the arguments to the argument parser. It takes no parameters directly, but gets
its arguments from the user-specified input
Returns: an ArgumentParser object with all the necessary arguments
'''
def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", type = int, metavar='n', help = "number of iterations", default = 100)
    parser.add_argument("-r", type = int, metavar = 'r', default = 214)
    parser.add_argument("drugs", type = str, help = "drug filename")
    parser.add_argument("targets", type = str, help = "targets filename")
    parser.add_argument('proteinA', type=str)
    parser.add_argument('proteinB', type=str)
    args = parser.parse_args()
    #print(str(args))
    return args
'''
This function sets the necessary variables using the ArgumentParser
Parameters:
    args: an ArgumentParser object
Returns: a tuple with the following variables, in the order below
    n: number of iterations for bootstrap p-value calculation (default is 100)
    r: random seed (default is 214)
    drugs: the filename of the drug file
    targets: the filename of the target file
    proteinA: the first protein for which we're calculating the p-value
    proteinB: the second protein for which we're calculating the p-value
''' 
def setArgs(args):
    n = args.n
    r = args.r
    drugs = args.drugs
    targets = args.targets
    proteinA = args.proteinA
    proteinB = args.proteinB
    return n, r, drugs, targets, proteinA, proteinB
'''
This program calculates the bootstrapped p-value of the Tanimoto Summary Score between two proteins. 
It takes the user-specified parameters (see above for descriptions) n, r, drugs, targets, proteinA, and proteinB. 
It then calculates the Tanimoto Summary Score of the similarity between the two proteins, and then a bootstrapped
p-value based on "n" random samplings of the ligands those proteins could sare. 
It prints the bootstrapped p-value to the console. 
'''   
def main():
    args = parseArgs()
    n, r, drugFile, targetFile, proteinA, proteinB = setArgs(args)
    targets = util.readTargetsData(targetFile)
    drugIDs, fingerprints = util.readDrugData(drugFile)
    ligands1 = util.getLigands(proteinA, targets)
    ligands2 = util.getLigands(proteinB, targets)
    num1 = len(ligands1)
    num2 = len(ligands2)
    real_score = util.tanimotoSummary(drugIDs, fingerprints, ligands1, ligands2)
    bootstrap_pval = util.calcBootstrap(n, real_score, num1, num2, drugIDs, fingerprints, r)
    print(str(bootstrap_pval))
    
if __name__ == "__main__":
    main()