import numpy as np
import random
import math
import matplotlib.pyplot as plt

pi = np.pi

movesPerDay = 20 #average moves per day
foodEnergy = 0.5
nTimeSteps = 10
size = 100
nSpeciesA = 10
nDays = 60
growthConstant = 0.1
timeSteps = np.arange(0, 2 * pi, 2 * pi / nTimeSteps)

class animal:
    energy = 10
    foodEaten = 0
    moveHistory = []
    
    def __init__(self, amp, phaseAngle):
        self.pos = [random.randint(0, size), random.randint(0, size)]
        self.amp = amp
        self.phaseAngle = phaseAngle

    def move(self):
        self.energy = self.energy - 1/movesPerDay
        self.pos = [random.randint(0, size), random.randint(0, size)]
            
    def checkAndEat(self):
        global foodList
        if self.pos in foodList:
            self.energy = self.energy + foodEnergy
            foodList.remove(self.pos) 
            self.foodEaten = self.foodEaten + 1
            
    def performMoves(self):   
        self.activeness = 1 + np.sin(t + self.phaseAngle) * self.amp #range 0 to 2       
        self.nMovesInStep = movesPerDay / nTimeSteps * self.activeness
        
        if (self.nMovesInStep % 1) > random.uniform(0,1):
            self.nMovesInStep = math.ceil(self.nMovesInStep)
        else:
            self.nMovesInStep = math.floor(self.nMovesInStep)       
        
        for r in range(0, self.nMovesInStep):
            self.move()
            self.checkAndEat()
     
            
    def sex(self, mate):
        parentAmp = (self.amp + mate.amp) / 2
        parentPhase = (self.phaseAngle + mate.phaseAngle) / 2 # need vector math
        childAmp = parentAmp + random.gauss(parentAmp, 0.03)
        if childAmp < 0:
            childAmp = 0
        if childAmp>1:
            childAmp = 1
            
        childPhase = parentPhase + random.gauss(parentPhase, 0.03*pi)
        if childPhase < 0:
            childPhase = 2*pi + childPhase
        if childPhase > 2*pi:
            childPhase = childPhase - 2*pi
        child = animal(childAmp, childPhase)
            
        return child
    
        
# initialize ecosystem
foodPerDay = 400
foodList = [[random.randint(0, size), random.randint(0, size)] for f in range(0, foodPerDay)]
foodAges = [random.randint(0,nTimeSteps) for f in range(0, foodPerDay)]

speciesA = []
for n in range(0, nSpeciesA):
    speciesA.append(animal(0, 0))
    

# Run ecosystem
foodPerStep = 2 * foodPerDay / nTimeSteps
foodOverTime = []
animalsOverTime = []

for d in range(0, nDays):
    
    for t in timeSteps:
        # food creation
        for f in range(0, int(foodPerStep)):
            foodList.append( [random.randint(0, size), random.randint(0, size)])
            foodAges.append(0)
        if  (foodPerStep % 1) > random.uniform(0, 1):
            foodList.append( [random.randint(0, size), random.randint(0, size)])
            foodAges.append(0)
        #food age
        for f in range(0, len(foodList)):
            foodAges[f] = foodAges[f]+1
        #food removal
        foodInds = [f for f in range(0, len(foodList)) if foodAges[f]!=nTimeSteps]
        foodList = [foodList[f] for f in foodInds]
        foodAges = [foodAges[f] for f in foodInds]  
            
        # animal loop
        for indiv in speciesA:
            indiv.performMoves()
            indiv.energy = indiv.energy - 1/nTimeSteps #basal metabolism
                
        #remove dead animals
        speciesA = [n for n in speciesA if n.energy>0]
        nSpeciesA = len(speciesA)
    
    #reproduction
    energies = np.array( [n.energy for n in speciesA])
    fitnesses = energies / sum(energies)
    fitnessesCumu = fitnesses
    for n in range(1, nSpeciesA):
        fitnessesCumu[n] = fitnessesCumu[n-1] + fitnesses[n]
    
    if (nSpeciesA * growthConstant % 1) > random.uniform(0,1):
        nNewAnimals = math.ceil(nSpeciesA * growthConstant)
    else:
        nNewAnimals = math.floor(nSpeciesA * growthConstant)  
            
    for n in range(0, nNewAnimals):
        ind1 = np.where(fitnessesCumu > random.uniform(0, 1))[0][0]
        ind2 = np.where(fitnessesCumu > random.uniform(0, 1))[0][0]
        
        speciesA.append( speciesA[ind1].sex( speciesA[ind2]))
  
    
    
    #report
    foodOverTime.append(len(foodList))
    animalsOverTime.append( len( speciesA))
    


#report
for n in range(0, nSpeciesA):
     indiv = speciesA[n]
     print(indiv.energy)

    

