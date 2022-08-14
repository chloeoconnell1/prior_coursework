import numpy as np
import argparse
from math import sqrt
from datetime import datetime

'''
This function parses the arguments and sets their defaults based on user input
Returns: an ArgumentParser object from the "argparse" module
The following are the arguments of the argParser object:
    iF = input filename
    kB = spring constant for bonds
    kN = spring const for non-bonds
    nbCutoff = non-bonding cutoff (distance above which atom interactions are ignored)
    m = mass of all atoms (assumed to be the same for all atoms)
    dt = timestep length
    n = number of timesteps to be simulated
    out = prefix of the output file
'''
def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iF", type = str)
    parser.add_argument("--kB", type = float, default = 40000.0)
    parser.add_argument("--kN", type = float, default = 400.0)
    parser.add_argument('--nbCutoff', type = float, default = 0.50)
    parser.add_argument('--m', type = float, default = 12.0)
    parser.add_argument('--dt', type = float, default =0.001)
    parser.add_argument('--n', type = int, default = 1000)
    parser.add_argument('--out', type = str)
    args = parser.parse_args()
    return args

'''
The setArgs sets the global variables (parameters) based on the argumentParser object's user-specified (or default) arguments
Inputs: args, the ArgumentParser objects containing the default and/or user-specified arguments
Returns: a tuple of variables:
    iF = input filename
    kB = spring constant for bonds
    kN = spring const for non-bonds
    nbCutoff = non-bonding cutoff (distance above which atom interactions are ignored)
    m = mass of all atoms (assumed to be the same for all atoms)
    dt = timestep length
    n = number of timesteps to be simulated
    out = prefix of the output file
'''
def setArgs(args):
    iF = args.iF
    kB = args.kB
    kN = args.kN
    nbCutoff = args.nbCutoff
    m = args.m
    dt = args.dt
    n = args.n
    if not args.out:
   	args.out = args.iF.split('.')[0]
    out = args.out
    return iF, kB, kN, nbCutoff, m, dt, n, out

'''
This function reads the initial positions of each atom from the input file
Inputs: fileName, the user-specified filename of the input file
Returns: posArray, an array consisting of the position of each atom
    - Rows = atoms
    - Columns = x, y, z positions
    - Note: The 0 row of this array is three zeros (never used) to allow for indexing by atom number
    (i.e. so that atom number 1 could be accessed using posArray[1, ])
'''
def initPos(fileName):
    positions = np.loadtxt(fileName, dtype = float, usecols = (1, 2, 3))
    zers = np.zeros(3)
    posArray = np.vstack([zers, positions])
    return posArray

'''
This function reads the initial velocities of each atom from the input file
Inputs: fileName, the user-specified filename of the input file
Returns: velArray an array specifying the x, y, and z velocities of each atom
    - Rows = atoms
    - Columns = x, y, and z velocities
    Note: the 0 row is again 0s (not used) so that velArray[1,] corresponds to the velocities of atom #1
'''
def initVel(fileName):
    vel = np.loadtxt(fileName, dtype = float, usecols = (4, 5, 6))
    zers = np.zeros(3)
    velArray = np.vstack([zers, vel])
    return velArray

'''
This function reads in the bonds from the input file
Inputs: fileName, the user-specified filename of the input file
Returns: 
    bonds: a list of lists, corresponding to the atoms bonded to each atom. For example, bonds[1]
is a list containing each atom atom #1 is bonded to 
'''
def initBonds(fileName):
    bonds = [[0]]
    lines = file(fileName).readlines()
    for line in lines[1:]:
        bonds.append([int(n) for n in line.split()[7:]])
    return bonds

'''
This function calculates the pairwise distances between all atoms
Inputs: positions, a matrix containing a row for each atom and columns denoting the x, y, and z coordinates of that atom
Returns: distMat, a matrix containing all pairwise distance between atoms
    - Again this matrix has a row and column of zeros as the first row/col so that distMat[1,2] corresponds to the distance btw atom 1 and atom 2
'''
def calcDistMat(positions):
    distMat = np.zeros((positions.shape[0], positions.shape[0]))
    # Loop through cols and rows, get distances
    # Only do it for i > j (i = row, j = col)
    for i in range(1, positions.shape[0]):
        for j in range(1, positions.shape[0]):
            if i == j:
                distMat[i, j] = 0
            elif i < j:
                pass
            else:
                distMat[i,j] = dist(i, j, positions)
                distMat[j,i] = distMat[i,j]
    return distMat

'''
This function calculates the euclidean distance between to atoms in the molecule
Inputs:
    atomnum1: the number of the first atom
    atomnum2: the number of the second atom
    positions: a matrix containing the current x, y, and z coordinates of all atoms in the molecule
Returns:
    the euclidean distance between the two atoms
'''
def dist(atomnum1, atomnum2, positions):
    pos1 = positions[atomnum1,]
    pos2 = positions[atomnum2,]
    sum = float(0)
    for i in range(len(pos1)):
	sum += (float(pos1[i]) - float(pos2[i]))**2
    return sqrt(sum)
    
'''
This function finds all atoms within the nonbonding cutoff from each other (at the start of the simulation)
Inputs:
    distanceMat: a matrix of pairwise distances between atoms
    bonds: a list of bonded atoms
Returns: 
    nbonds, a list of lists representing the atoms within non-bonding distance to each atom
        (each index of nbonds represents an atom, and within that index the elements of the list represent the nearby atoms)
'''
def findnb(distanceMat, bonds):
    nbonds = [[0]]
    for i in range(1, distanceMat.shape[0]):
        nb = []
        for j in range(1, distanceMat.shape[1]):
            if  i == j or j in bonds[i]:
                pass
            else:
                if (distanceMat[i, j] < args.nbCutoff) and (i != j):
                    nb.append(j)
        nbonds.append(nb)
    return nbonds

'''
This function updates the positions based on the Velocity Verlet algorithm
Inputs: 
    positions: a matrix representing the x, y, and z coordinates of each atom to be updated (i.e. previous positions) 
    vthalf: a matrix representing the x, y, and z velocities of each atom at t+dt/2
Returns: a new position matrix representing the x, y, and z coordinates of each atom at t + dt/2
'''
def updatePositions(positions, vthalf):
    deltaR = vthalf*dt
    newPos = np.add(deltaR, positions)
    return newPos
 
'''
This function calculates the potential energy at a given timestep coming from bond energies
Inputs: 
    newDistances: a matrix representing the current pairwise distances between atoms 
        (note: only the bonded and non-bonded atoms' distances are updated at each timestep to save time/computation power)
    positions: a matrix representing the current positions of each
Returns: PE, the total potential energy of all bonds
'''
def calcPEbond(newDistances, positions):
    PE = 0.0
    for atomNum in range(1, len(bonds)):
        for bondedAtomIndex in range(len(bonds[atomNum])):
            bondedAtomNum = bonds[atomNum][bondedAtomIndex]
            if atomNum < bondedAtomNum:
                bondEnergy = calcBondEnergy(atomNum, bondedAtomNum, newDistances, positions)
                PE += bondEnergy
    return PE

'''
This function calculates the potential energy of an individual bond, while updating the force matrix keeping track of the force on each atom
Inputs:
    atomNum1: the index of the first atom
    atomNum2: the index of the second atom
    newDistances: a matrix representing the current pairwise distances between atoms 
    positions: a matrix representing the current positions of each atom
Returns:
    bondEnergy: the bond energy of the bond between atomNum1 and atomNum2
    (* also updates the force matrix (global variable forceMat), factoring that bond into the forces acting on atomNum1 and atomNum2)
'''
def calcBondEnergy(atomNum1, atomNum2, newDistances, positions): # calculate PE given old and new distances, also update forceMat
    newDist = float(newDistances[atomNum1, atomNum2])
    optimalDist= float(initDistances[atomNum1, atomNum2])
    bondEnergy = 0.5*kB*((newDist - optimalDist)**2)
    netForce = kB*(newDist - optimalDist)
    #print(str(positions[atomNum2, ]))
    Fx = netForce * (positions[atomNum2, 0] - positions[atomNum1,0])/newDist
    Fy = netForce * (positions[atomNum2, 1] - positions[atomNum1,1])/newDist
    Fz = netForce * (positions[atomNum2, 2] - positions[atomNum1,2])/newDist
    forceMat[atomNum1,] = forceMat[atomNum1,] + [Fx, Fy, Fz,]
    forceMat[atomNum2,] = forceMat[atomNum2,] - [Fx, Fy, Fz,]
    return bondEnergy

'''
This function calculates the potential energy of a non-bonding interaction between two atoms, while updating the force matrix (global variable) 
keeping track of the force (in the x, y, and z directions) on each atom
Inputs:
    atomNum1: the index of the first atom
    atomNum2: the index of the second atom
    newDistances: a matrix representing the current pairwise distances between atoms 
    positions: a matrix representing the current positions of each atom
Returns:
    nbEnergy: the non-bonding energy of the bond between atomNum1 and atomNum2
    (* also updates the force matrix (global variable forceMat), factoring that bond into the forces acting on atomNum1 and atomNum2)
'''
def calcNbEnergy(atomNum1, atomNum2, newDistances, positions): # should now also update forces AND calculate non-bond energy
    newDist = float(newDistances[atomNum1, atomNum2])
    optimalDist= float(initDistances[atomNum1, atomNum2])
    nbEnergy = 0.5*kN*((newDist - optimalDist)**2)
    netForce = kN*(newDist - optimalDist)
    Fx = netForce * (positions[atomNum2, 0] - positions[atomNum1,0])/newDist
    Fy = netForce * (positions[atomNum2, 1] - positions[atomNum1,1])/newDist
    Fz = netForce * (positions[atomNum2, 2] - positions[atomNum1,2])/newDist
    forceMat[atomNum1,] = forceMat[atomNum1,] + [Fx, Fy, Fz,]
    forceMat[atomNum2,] = forceMat[atomNum2,] - [Fx, Fy, Fz,]
    return nbEnergy
    
'''
This function calculates the potential energy at a given timestep coming from non-bonding interactions
Inputs: 
    newDistances: a matrix representing the current pairwise distances between atoms 
        (note: only the bonded and non-bonded atoms' distances are updated at each timestep to save time/computation power)
    positions: a matrix representing the current positions of each
Returns: PE, the total potential energy from all non-bonding interactions
'''
def calcPEnb(newDistances, positions):
    PE = 0.0
    for atomNum in range(1, len(nB)):
        for nbAtomIndex in range(len(nB[atomNum])):
            nbAtomNum = nB[atomNum][nbAtomIndex]
            if atomNum < nbAtomNum:
                nbEnergy = calcNbEnergy(atomNum, nbAtomNum, newDistances, positions)
                PE += nbEnergy
    return PE
    
'''
This function updates the total kinetic energy based on the velocities of all atoms
Inputs: velocities, the matrix of x, y, and z velocities of each atom
Returns: KEtot, a float representing the total kinetic energy of all atoms
'''
def updateKE(velocities):
    vsquared = np.square(velocities)
    KEmat = 0.5*m*vsquared
    KEtot = np.sum(KEmat)
    return KEtot

'''
This function accesses the header for the top of the .rvc output file (from the input file)
'''
def getRvcHeader():
    with open(iF, 'r') as f:
        header = f.readline()
    return str(header)

'''
This function writes the positions and velocities of each atom to the .rvc file after each timestep
Inputs: 
    positions: the matrix of current x, y, z coordinates of each atom
    velocities: the matrix ofcurrent x, y, z velocities of each atom
    out_rvc: the filename of the .rvc output file
    bonds: the list of lists, containing lists of atoms each individual atom is bonded to
'''
def writeRVC(positions, velocities, out_rvc, bonds):
    posVel = np.concatenate((positions, velocities), axis = 1)
    for atom in range(1, velocities.shape[0]):
        out_rvc.write(str(atom) + "\t" + '\t'.join(map("{:.4f}".format, posVel[atom,])) + "\t" + '\t'.join(map(str, bonds[atom]))+ "\n")

'''
This function updates relevant distances (i.e. distances between all bonded and non-bonded atoms) in the pairwise distance matrix
Note: distanceMat is a global variable, so it is modified in place
This function also only updates pairwise distances involved in bonding and non-bonding interactions, because
originally I had coded updates on all distances and the program was insanely slow
Inputs:
    positions: the matrix containing x, y, and z coordinates of each atom
'''
def updateDistanceMat(positions):
    for atomNum in range(1, len(nB)):
        for nbAtomIndex in range(len(nB[atomNum])):
            nbAtomNum = nB[atomNum][nbAtomIndex]
            if atomNum < nbAtomNum:
                #print("nonbond atoms = %s")%str((atomNum, nbAtomNum))
                distance = dist(atomNum, nbAtomNum, positions)
                distanceMat[atomNum, nbAtomNum] = distance
                distanceMat[nbAtomNum, atomNum] = distance
    for atomNum in range(1, len(bonds)):
        for bondAtom in range(len(bonds[atomNum])):
            bondAtomNum = bonds[atomNum][bondAtom]
            if atomNum < bondAtomNum:
                #print("bond atoms = %s")%str((atomNum, bondAtomNum))
                distance = dist(atomNum, bondAtomNum, positions)
                distanceMat[atomNum, bondAtomNum] = distance
                distanceMat[bondAtomNum, atomNum] = distance

'''
This is the main function. The overall algorithm is as follows:
    1. declare global variables, read in arguments
    2. sets global variables for each of the parameters based on arguments
    3. Initialize positions, velocities and bond interactions based on inputs
        - Compute distance matrix with pairwise distances between all atoms
        - Identify non-bonding interactions based on distance matrix
    4. For each timestep... 
        - Updates the velocities (for t+dt/2) on each atom
        - For each interaction pair (bonded and non-bonded):
            - Calculates the potential energy and updates the total potential energy
            - Calculates the force and updates the total forces in each dimension for each atom
            - Updates the velocities (for t+dt) on each atom and calculate the kinetic energy
        - Writes energy output every 10 timestep to the .erg output file, and positional/velocity info to the .rvc output file every timestep
'''
def main():
    global iF
    global kB
    global kN
    global nbCutoff
    global m
    global dt
    global out
    global initDistances # Matrix with the reference distances for each bond/nonBond pair
    global bonds # list of lists designating which atoms are bonded
    global nB # List of lists designating which atoms are "non-bonded"
    global forceMat
    global distanceMat
    global args
    args = parseArgs()
    iF, kB, kN, nbCutoff, m, dt, n, out = setArgs(args)   
    out_erg = open(out + "_out.erg","w")
    out_rvc = open(out + "_out.rvc", "w")
    out_erg.write("# step\tE_k\tE_b\tE_nB\tE_tot\n")
    out_rvc.write(getRvcHeader())
    
    positions = initPos(iF)
    velocities = initVel(iF)
    
    initDistances = calcDistMat(positions) 
    distanceMat = np.copy(initDistances)
    # distanceMat[1,2] is dist btw 1 and 2
    bonds = initBonds(iF)
    nB = findnb(initDistances, bonds)
    forceMat = np.zeros((initDistances.shape[0], 3))
    
    writeRVC(positions, velocities, out_rvc, bonds)
    
    for timestep in range(1, n + 1):
        accel = forceMat / m
        vthalf = velocities + (0.5 * accel * dt)
        
        ## Update positions
        updatedPositions = updatePositions(positions, vthalf)
        positions = updatedPositions
        updateDistanceMat(positions)
        newDistances = distanceMat
        
        # we're going to update the force matrix, so we need to zero it out first:
        forceMat = np.zeros((initDistances.shape[0], 3))
        
        # Now calculate PE and update force matrix
        PEbond = calcPEbond(newDistances, positions) # bonds, initDistances, 
        PEnonbond = calcPEnb(newDistances, positions)
        PEtot = PEbond + PEnonbond
        
        # Now update velocities again and calculate KE
        atdt = forceMat / m
        velocities = vthalf + (0.5 * atdt * dt)
        KEtot = 0
        KEtot = updateKE(velocities)
        
        # Check for overflow
        if timestep == 1:
            startingEnergy = PEtot + KEtot
        if PEtot + KEtot >= (startingEnergy*10):
            print("Overflow energy! Quitting")
            break
        if timestep % 10 == 0:
            totalEnergy = KEtot + PEtot
            out_erg.write(str(timestep) + "\t" + str(round(KEtot, 1)) + "\t" + str(round(PEbond, 1)) + "\t" + str(round(PEnonbond, 1)) + "\t" + str(round(totalEnergy, 1))+"\n")
            out_rvc.write("#At time step " + str(timestep) + ",energy = " + str(round(totalEnergy, 3)) + "kJ \n")
            writeRVC(positions, velocities, out_rvc, bonds)

main() 
    

