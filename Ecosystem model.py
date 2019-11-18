import numpy as np
import random
import math
import scipy.stats
import scipy.cluster
import matplotlib.pyplot as plt

pi = np.pi

nTimeSteps = 10
popStabilization = True
assortativeMating = True
species = ('A')
ampInit = {'A': 1, 'B': 1}
phaseInit = {'A': pi, 'B': pi}
ampStd = 0
phaseStd = 0.04
nAnimalsInit = {'A': 50, 'B': 50}
growthRatesAll = {'A': 0.1, 'B': 0.1}
matingWindow = 9/24
nDays = 1000
nTrials = 1
arenaSize = 60 #was 50
foodPerDay = arenaSize * 10
specialization = 0.0
energyInit = 10
mutantProb = 0
lifespan = 12
movesPerDay = 20 #average moves per day
foodEnergy = 0.5

timeSteps = np.arange(0, 2*pi, 2*pi / nTimeSteps)

# create referencee activity profile
nMiniSteps = 10000
nMiniStepsPerStep = int( nMiniSteps / nTimeSteps)
miniSteps = np.arange( 0, 2*pi, 2*pi/nMiniSteps)
rawActProfile = (1 + np.cos( miniSteps)) ** 1
actProfileSum = sum( rawActProfile) / nMiniSteps * 2*pi  #intrgal of activity profile
actProfile = rawActProfile * movesPerDay / actProfileSum  #normalized so that integral = movesPerDay
meanActivity = np.mean( actProfile) #for later calculations


class animal:
    foodEaten = 0
    age = 0
    activeness = 0
    energyPerMove = 1/movesPerDay
    
    def __init__( self, species, amp, phase, energy):
        self.loc = random.randint(0, arenaSize)
        self.species = species
        self.amp = amp
        self.phase = phase
        self.energy = energy
        
    def move( self):
        self.energy = self.energy - self.energyPerMove
        self.loc = random.randint(0, arenaSize)
            
    def computeNumMoves( self):               
        # calculate number of moves to perform this step based on activity profile
        self.actIndStart = int( (t - self.phase) / (2*pi)*nMiniSteps) #where on the activity profile
        if not (0 <= self.actIndStart < nMiniSteps):
            self.actIndStart = self.actIndStart % nMiniSteps 
        
        self.nMovesInStep = np.mean( actProfile[self.actIndStart : self.actIndStart + nMiniStepsPerStep])
        self.nMovesInStep = meanActivity + (self.nMovesInStep - meanActivity) * self.amp #adjusted for amplitude
        
        if (self.nMovesInStep % 1) > random.uniform(0,1):
            self.nMovesInStep = math.ceil(self.nMovesInStep)
        else:
            self.nMovesInStep = math.floor(self.nMovesInStep)       
#        
#        for r in range(0, self.nMovesInStep):
#            self.move()
#            self.checkAndEat()
            
    def checkAndEat( self):
        global foodLocs
    
        if self.loc in foodLocs:               
            self.energyGained = (1 + np.cos( self.phase - t) * specialization) * foodEnergy            
            self.energy = self.energy + self.energyGained
            foodLocs.remove(self.loc) 
            self.foodEaten = self.foodEaten + 1
            
    def sex( self, mate):
        childAmp = random.gauss((self.amp + mate.amp)/2, ampStd)
        if childAmp < 0:
            childAmp = 0
        if childAmp > 1:
            childAmp = 1   
            
        childPhase = random.gauss( scipy.stats.circmean([self.phase, mate.phase]),
                                  2*pi*phaseStd)
        if not (0 <= childPhase < 2*pi):
            childPhase = childPhase % (2*pi)

        #mutations
        if random.uniform(0,1) < mutantProb:
            childAmp = random.uniform(0,1)
            childPhase = random.uniform(0,2*pi)
            
        childEnergy = self.energy/3 + mate.energy/3
        
        child = animal(self.species, childAmp, childPhase, childEnergy)
            
        return child
    
    
## Mega Trial loop
allPhaseA = []
allPhaseB = []
allCrossGroupMean = []

output = np.zeros([nTrials, 4], dtype='int')

for trial in range(0, nTrials):
    print(trial)
    
    # Initialize ecosystem   
    animals = []
    for s in species:
        animals = animals + [animal(s, ampInit[s], phaseInit[s] + random.gauss(0, 1.5*phaseStd), 
                                    random.uniform(0,energyInit)) for n in range(0,nAnimalsInit[s])]
        
#        animals = animals + [animal(s, ampInit[s], phaseInit[s]+pi, random.uniform(0,energyInit)) for 
#                         n in range(0,nAnimalsInit[s])]
#    single species    
#    animals = animals + [animal('A', 1, 0, random.uniform(0, energyInit)) for n in range(0, 30)]
#    animals = animals + [animal('A', 1, pi, random.uniform(0, energyInit)) for n in range(0, 175)]
    
    random.shuffle(animals)
   
    for indiv in animals:
        indiv.age = random.uniform(0, lifespan)    
    
    foodLocs = list(range(0,100))
    foodPerStep = foodPerDay / nTimeSteps
    foodOverTime = []
    popOverTimeA = [0] * nDays
    popOverTimeB = [0] * nDays
    activenessA = []
    phaseStdOverTimeA = []
    phaseStdOverTimeB = []
    phaseOverTimeA = []
    phaseOverTimeB = []
    ampOverTimeA = []
    ampOverTimeB = [] 
    ampStdOverTimeA = []
    ampStdOverTimeB = []
    crossGroupMeanOverTime = []
    scatterEnergies = []
    scatterPhases = []
    
    # Run ecosystem
    for d in range(0, nDays):
        
        for t in timeSteps:                           
            # movements, food creation, basal metabolism, aging
            movesList = []
            for n in range( 0, len(animals)):
                indiv = animals[n]
                
                indiv.energy -= 1/nTimeSteps #basal metabolism
                indiv.age += 1/nTimeSteps
                
                indiv.computeNumMoves()
                movesList.extend( np.repeat( n, indiv.nMovesInStep))
            
            random.shuffle( movesList)
            
            for n in movesList:                
                animals[n].move()
                animals[n].checkAndEat()
                
                # food creation interspersed with movements for more evenness             
                foodToMake = foodPerStep / len(movesList)
                quotient = int( foodToMake // 1)
                remainder = foodToMake % 1
                
                if foodToMake >= 1:
                    foodLocs = list( range( 0, len(foodLocs) + quotient))
                    
                if random.uniform(0,1) < remainder:
                    foodLocs = list( range( 0, len(foodLocs) + 1))
                           
                if len(foodLocs) > 0:
                    if foodLocs[-1] > arenaSize-1:
                        foodLocs= list( range(0, arenaSize))
                                
                    
            # remove dead animals
            animals = [a for a in animals if a.energy>0.000001]        
            animals = [a for a in animals if a.age<lifespan-0.000001]
            nAnimals = len(animals)
        
        
            # reproduction
            for s in species:                
                speciesPop = [n for n in animals if n.species==s]
                nThisSpecies = len(speciesPop)
                animals = [a for a in animals if a.species!=s] #take out species s for the moment
                energies = [a.energy for a in speciesPop]
             
                if nThisSpecies < 2: #skip if not at least two available parents
                    continue
                
                if any([popStabilization==False, len(species)==1]): 
                    growthRate = growthRatesAll[s]
                else:
                    growthRate = growthRatesAll[s] * nAnimals / nThisSpecies / len(species)
                
                avgNewAnimals = ((1+growthRate)**(1/nTimeSteps) - 1) * nThisSpecies
                if (avgNewAnimals % 1) > random.uniform(0, 1):
                    nNewAnimals = math.ceil(avgNewAnimals)
                else:
                    nNewAnimals = math.floor(avgNewAnimals)                 
                                                
                # if assortative mating, create phase overlap and mating probability matrices               
                if assortativeMating == True:   
                    overlapMat = np.empty([nThisSpecies, nThisSpecies])
                    
                    # mating overlaps based on phase difference
                    for m in range( 0, nThisSpecies):
                        for n in range( m + 1, nThisSpecies):    
                            phaseDif = abs( speciesPop[m].phase - speciesPop[n].phase)
                            if phaseDif > pi:
                                phaseDif = 2*pi - phaseDif 
                            overlapMat[m,n] = overlapMat[n,m] = 1 - phaseDif / (2*pi * matingWindow)
                    
                    np.fill_diagonal(overlapMat, 0)
                    overlapMat[overlapMat < 0] = 0
                    
                    # mating overlaps based on full activity profiles
#                    actProfiles = [1 + np.sin(timeSteps + n.phase) * n.amp  for n in speciesPop]
#                    maxOverlap = sum((1+np.sin(timeSteps)) * (1+np.sin(timeSteps)))              
#                    
#                    for m in range(0, nThisSpecies):
#                        for n in range(0, nThisSpecies):
#                            overlapMat[m,n] = (sum( actProfiles[m] * actProfiles[n]) / maxOverlap)
                            
                    mateProbMat = overlapMat * [a.energy for a in speciesPop]
                    mateProbMat = np.matrix.transpose( mateProbMat) * [a.energy for a in speciesPop]
                    mateProbMat = mateProbMat / np.sum( np.triu(  mateProbMat))
                    triangleInds = np.triu_indices(nThisSpecies, k=1)
                            
                # select parents and mate                
                for a in range(0, nNewAnimals):                  
                    if assortativeMating == True:                                         
                        ind = np.random.choice( range( 0, len(triangleInds[0])), p=mateProbMat[triangleInds])
                        ind0 = triangleInds[0][ind]
                        ind1 = triangleInds[1][ind]                                                                                     
                    else:  #no assortative mating                                                
                        mateProb = np.array(energies) / sum(energies)                        
                        ind0 = np.random.choice( range(0, nThisSpecies), p=mateProb)       
                        mateProb[ind0] = 0 #ensure parents are not same individual
                        mateProb = mateProb / sum( mateProb)
                        ind1 = np.random.choice( range(0, nThisSpecies), p=mateProb)  
                                       
                    speciesPop.append( speciesPop[ind0].sex( speciesPop[ind1]))
                    speciesPop[ind0].energy = speciesPop[ind0].energy * 2/3
                    speciesPop[ind1].energy = speciesPop[ind1].energy * 2/3
                    
                animals = animals + speciesPop
            random.shuffle(animals)
            
            
#           Timestep-level records
#            activenessA.append(np.mean([n.activeness for n in animals if n.species=='A']))
            foodOverTime.append( len( foodLocs))


#       Day-level records
#        popOverTimeA[d] = len([1 for n in animals if n.species=='A'])
#        popOverTimeB[d] = len([1 for n in animals if n.species=='B'])
        phaseOverTimeA.append( scipy.stats.circmean([n.phase for n in animals if n.species=='A']))
        phaseOverTimeB.append( scipy.stats.circmean([n.phase for n in animals if n.species=='B']))
#        phaseStdOverTimeA.append( scipy.stats.circstd([n.phase for n in animals if n.species=='A']))
#        phaseStdOverTimeB.append( scipy.stats.circstd([n.phase for n in animals if n.species=='B']))
#        ampOverTimeA.append( np.mean([n.amp for n in animals if n.species=='A']))
#        ampOverTimeB.append( np.mean([n.amp for n in animals if n.species=='B']))
#        ampStdOverTimeA.append( np.std([n.amp for n in animals if n.species=='A']))
#        ampStdOverTimeB.append( np.std([n.amp for n in animals if n.species=='B']))
#        
#        if any([len([1 for n in animals if n.species=='A'])==0, len([1 for n in animals if n.species=='B'])==0]):
#            break
        
        #  Show figures for live updates
        if (d+1) % 100 == 0:
            print(trial, d)
            plt.figure()
            
#            plt.xlim([0, 2*pi])
            plt.hist([n.phase for n in animals if n.species=='A'], color=[0,0,1,0.5], bins=20)
#            plt.hist([n.phase for n in animals if n.species=='B'], color=[1,.64,0,0.5], bins=20)
            
#            plt.plot(ampOverTimeA, color='blue')
#            plt.plot(ampOverTimeB, color='orange')
            
#            plt.plot(phaseOverTimeA[(d-99): d], ampOverTimeA[(d-99): d], color='blue')
#            plt.plot(phaseOverTimeB[(d-99): d], ampOverTimeB[(d-99): d], color='orange')            
#            plt.xlim([0,2*pi])
#            plt.ylim([0,1]) 
            
            plt.show()
            
            
#        cluster mating matrix for measuring speciation
        if assortativeMating == True:
            cluster = scipy.cluster.vq.kmeans( scipy.cluster.vq.whiten( overlapMat), 2)
            groupClasses = [n[0] > n[1] for n in np.transpose(cluster[0])]
            groupClasses = np.array([int(n) for n in groupClasses])
            
            grps00_mat = overlapMat[np.ix_(groupClasses==0, groupClasses==0)]
            np.fill_diagonal(grps00_mat, np.nan)
            grps00_mean = np.nanmean(grps00_mat)
            
            grps11_mat = overlapMat[np.ix_(groupClasses==1, groupClasses==1)]
            np.fill_diagonal(grps11_mat, np.nan)
            grps11_mean = np.nanmean(grps11_mat)
            
            inGroupMean = (grps00_mean * sum(groupClasses==0) + grps11_mean*sum(groupClasses==1)) / len(groupClasses) 
            
            grps01_mat = overlapMat[np.ix_(groupClasses==0, groupClasses==1)]
            crossGroupMean = np.nanmean(grps01_mat)
            
            crossGroupMeanOverTime.append(crossGroupMean)
#        
            
#       energy by phase data
#        if (d+1) % 2 == 0:
#            scatterEnergies = scatterEnergies + [n.energy for n in animals]
#            meanPhase = scipy.stats.circmean([n.phase for n in animals])
#            buf = [n.phase - meanPhase for n in animals]
#            for n in range(0, len(buf)):
#                if buf[n] > pi:
#                    buf[n] = buf[n] - 2*pi
#                if buf[n] < -pi:
#                    buf[n] = buf[n] + 2*pi
#
#            scatterPhases = scatterPhases + buf
    
#   Trial-level records
#    plt.figure()
#    plt.plot(popOverTimeA)
#    plt.plot(popOverTimeB)    
#    plt.pause(0.0001)  
#    
#    allPopA.append(popOverTimeA)
#    allPopB.append(popOverTimeB)
#    allPhaseA.append(phaseOverTimeA)
#    allPhaseB.append(phaseOverTimeB)
#    allAmpA.append(ampOverTimeA)
#    allAmpB.append(ampOverTimeB)
    
#    allCrossGroupMean.append(crossGroupMeanOverTime)

    
    
## Write and finish
np.savetxt('C:/Users/Vance/Documents/Turek Lab/Temporal niche modeling/Amplitude 2 species spec0 - A.csv',
           np.transpose(allPopA), fmt="%.3f", delimiter=",")
np.savetxt('C:/Users/Vance/Documents/Turek Lab/Temporal niche modeling/Amplitude 2 species spec0 - B.csv',
           np.transpose(allPopB), fmt="%.3f", delimiter=",")

import time
print( time.ctime())