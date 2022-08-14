import numpy as np
import argparse
import sys
from math import sqrt


parser = argparse.ArgumentParser()
parser.add_argument("--iF", type=str, help = "Input filename")
parser.add_argument("--kB", type=float, default = 40000.0, help = "spring constant for bonds, default = 4000")
parser.add_argument("--kN", type=float, default = 400.0, help="spring constant for non-bonds, default = 400")
parser.add_argument('--nbCutoff', help='non-bonding cutoff', type=float, default='0.50')
parser.add_argument('--m', help='atom mass', type=float, default='12.0')
parser.add_argument('--dt', help='time step', type=float, default='0.001')
parser.add_argument('--n', help ='number of time steps', type=int, default='1000')
parser.add_argument('--out', help='prefix of the output filenames', type=str)

args = parser.parse_args()

def setArgs():
    iF = args.iF
    kB = args.kB
    kN = args.kN
    nbCutoff = args.nbCutoff
    m = args.m
    dt = args.dt
    n = args.n
    if not args.out:
   	args.out = args.iF.split('.')[0] + "_out"
    out = args.out
    return iF, kB, kN, nbCutoff, m, dt, n, out

def initPos(fileName):
    positions = np.loadtxt(fileName, dtype = float, usecols = (1, 2, 3))
    zers = np.zeros(3)
    posArray = np.vstack([zers, positions])
    return posArray

def initVel(fileName):
    vel = np.loadtxt(fileName, dtype = float, usecols = (4, 5, 6))
    zers = np.zeros(3)
    velArray = np.vstack([zers, vel])
    return velArray

# A list of list, 1-indexed so that bonds[1] corresponds to the bonds of 1
def initBonds(fileName):
    bonds = [[0]]
    lines = file(fileName).readlines()
    for line in lines[1:]:
        bonds.append([int(n) for n in line.split()[7:]])
    return bonds
    
def calcDistMat(positions):
    distMat = np.zeros((positions.shape[0], positions.shape[0]))
    # Loop through cols and rows, get distances
    # Only do it for i > j (i = row, j = col)
    for i in range(1, positions.shape[0]):
        for j in range(1, positions.shape[0]):
            if i == j:
                distMat[i, j] = 0
            else:
                distMat[i,j] = dist(i, j, positions)
    return distMat

def dist(atomnum1, atomnum2, positions):
    pos1 = positions[atomnum1,]
    pos2 = positions[atomnum2,]
    sum = float(0)
    for i in range(len(pos1)):
	sum += (float(pos1[i]) - float(pos2[i]))**2
    return sqrt(sum)
    
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

def updatePositions(positions, vthalf):
    deltaR = vthalf*dt
    newPos = np.add(deltaR, positions)
    return newPos
 
def calcPEbond(newDistances, positions):
    PE = 0.0
    for atomNum in range(1, len(bonds)):
        for bondedAtomIndex in range(len(bonds[atomNum])):
            bondedAtomNum = bonds[atomNum][bondedAtomIndex]
            if atomNum < bondedAtomNum:
                bondEnergy = calcBondEnergy(atomNum, bondedAtomNum, newDistances)
                PE += bondEnergy
                calcBondForce(atomNum, bondedAtomNum, newDistances, positions)
    return PE

def calcBondForce(atomNum1, atomNum2, newDistances, positions):
    newDist = float(newDistances[atomNum1, atomNum2])
    optimalDist= float(initDistances[atomNum1, atomNum2])
    netForce = kB*(newDist - optimalDist)
    #print(str(positions[atomNum2, ]))
    Fx = netForce * (positions[atomNum2, 0] - positions[atomNum1,0])/newDist
    Fy = netForce * (positions[atomNum2, 1] - positions[atomNum1,1])/newDist
    Fz = netForce * (positions[atomNum2, 2] - positions[atomNum1,2])/newDist
    forceMat[atomNum1,] = forceMat[atomNum1,] + [Fx, Fy, Fz,]
    forceMat[atomNum2,] = forceMat[atomNum2,] - [Fx, Fy, Fz,]
    
def calcNbForce(atomNum1, atomNum2, newDistances, positions):
    newDist = float(newDistances[atomNum1, atomNum2])
    optimalDist= float(initDistances[atomNum1, atomNum2])
    netForce = kN*(newDist - optimalDist)
    Fx = netForce * (positions[atomNum2, 0] - positions[atomNum1,0])/newDist
    Fy = netForce * (positions[atomNum2, 1] - positions[atomNum1,1])/newDist
    Fz = netForce * (positions[atomNum2, 2] - positions[atomNum1,2])/newDist
    forceMat[atomNum1,] = forceMat[atomNum1,] + [Fx, Fy, Fz,]
    forceMat[atomNum2,] = forceMat[atomNum2,] - [Fx, Fy, Fz,]
    
def calcPEnb(newDistances, positions):
    PE = 0.0
    for atomNum in range(1, len(nB)):
        for nbAtomIndex in range(len(nB[atomNum])):
            nbAtomNum = nB[atomNum][nbAtomIndex]
            if atomNum < nbAtomNum:
                nbEnergy = calcNbEnergy(atomNum, nbAtomNum, newDistances)
                PE += nbEnergy
                calcNbForce(atomNum, nbAtomNum, newDistances, positions)
    return PE

def calcBondEnergy(atomNum1, atomNum2, newDistances): # calculate PE given old and new distances
    newDist = float(newDistances[atomNum1, atomNum2])
    optimalDist= float(initDistances[atomNum1, atomNum2])
    bondEnergy = 0.5*kB*((newDist - optimalDist)**2)
    return bondEnergy
    
def calcNbEnergy(atomNum1, atomNum2, newDistances):
    newDist = float(newDistances[atomNum1, atomNum2])
    optimalDist= float(initDistances[atomNum1, atomNum2])
    nbEnergy = 0.5*kN*((newDist - optimalDist)**2)
    return nbEnergy

def updateKE(velocities):
    v2 = np.square(velocities)
    KEmat = 0.5*m*v2
    KEtot = np.sum(KEmat)
    return KEtot
    

def main():
    global kB
    global kN
    global nbCutoff
    global m
    global dt
    global n
    global out
    global initDistances # Matrix with the reference distances for each bond/nonBond pair
    global bonds # list of lists designating which atoms are bonded
    global nB # List of lists designating which atoms are "non-bonded"
    global forceMat
    iF, kB, kN, nbCutoff, m, dt, n, out = setArgs()    
    
    positions = initPos(iF)
    velocities = initVel(iF)
    initDistances = calcDistMat(positions) 
    # distanceMat[1,2] is dist btw 1 and 2
    bonds = initBonds(iF)
    nB = findnb(initDistances, bonds)
    #print(str(nB))
    forceMat = np.zeros((initDistances.shape[0], 3))
    
    for timestep in range(1,100):
        accel = forceMat / m
        vthalf = velocities + (0.5 * accel * dt)
        
        ## Update positions
        updatedPositions = updatePositions(positions, vthalf)
        positions = updatedPositions
        newDistances = calcDistMat(positions)
        
        # we're going to update the force matrix, so we need to zero it out first:
        forceMat = np.zeros((initDistances.shape[0], 3))
        
        # Now calculate PE and update force matrix
        PEbond = calcPEbond(newDistances, positions) # bonds, initDistances, 
        PEnonbond = calcPEnb(newDistances, positions)
        PEtot = PEbond + PEnonbond
        
        # Now update velocities again and calculate KE
        atdt = forceMat / m
        updateVel = vthalf + (0.5 * atdt * dt)
        velocities = updateVel
        KEtot = 0
        KEtot = updateKE(velocities)
        if timestep % 10 == 0:
            print("KEtot =")
            print(str(KEtot))
            print("PEtot")
            print(str(PEtot))
            print("Total energy = ")
            totalEnergy = KEtot + PEtot
            print(str(totalEnergy))
        
    ''' 
    For each iteration
        update velocities based on force matrix
        Update positions based on velocities
    ''' 


if __name__=="__main__":
    main() 
    
    '''
    dictionaries/arrays:
        positions: array of x, y, z positions
        velocities: array of x, y, z velocities
        forces: array of x, y, z forces (should be calculated)
        distance matrix: n by n matrix (atoms by atoms) of distances (only fill in half)
        bond list = list of lists representing bonds (blist[1] would be a list of atoms atom 1 is bonded to)
        nblist = list of lists representing atoms < nbCutoff (nblist[1] would be all the atoms within nbCutoff from 1)
Steps
    - Initialize positions, beginning forces, and velocities = 0
    - Create distance matrix, bond and non-bond list
    - For each time step:
        - Update velocities (for t+dt/2) based on previous force matrix (starts at 0, nx3 matrix)
        - Update positions based on velocities: r(t + dt) = r(previous) + [v(previous) + 1/2 a(prev) * dt]dt
        - Reset total energy to 0
        - For all bonded and non-bonded interactions
            - Calculate PE, use this to update
                - Only add to energy if j > i (don't want to add something twice
            - Calculate force, update forces on each atom
                ## THIS will be hard
        - Update velocities again (for t + dt) and calculate KE
    - If time % 10 == 0, write to file
    '''



