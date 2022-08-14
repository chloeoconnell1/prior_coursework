import sys
import numpy as np
import chemoUtils as util

'''
This program calculates all tanimoto scores of all drug pairs using the "allScores" function from the
chemoUtils program. 
Parameters: 
    First argument: file name of the drug input file, which contains 3 columns: db_id, generic_name, maccs (aka space-delimited features)
    Second argument: file name of the targets input file, a csv containing 3 columns: db_id, uniprot_accession, uniprot_id
    Third argument: output filename, to which the program writes the tanimoto scores of all drug pairs
Output file: 
    Column 1: drugbank ID #1
    Column 2: drugbank ID #2
    Column 3: Tanimoto score (6 decimal places) between the two
    Column 4: 1 or 0 indicating whether the drugs share a target or not (1 = target shared)
'''
def main():
    scoreMatrix = util.allScores(sys.argv[1],sys.argv[2])
    outFile = sys.argv[3]
    np.savetxt(outFile, scoreMatrix,delimiter=',', fmt=('%s', '%s', '%s', '%s'))
    
if __name__ == "__main__":
    main()