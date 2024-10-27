import numpy as np
import matplotlib.pyplot as plt 
import scipy.io as sio 
import bitarray as bta 
import bitarray.util as butil 
import transforms as tr 
import time 


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.preprocessing import normalize
import time 

#Quantum computing 
from qiskit import QuantumCircuit 
from qiskit import QuantumRegister 
from qiskit import ClassicalRegister 
from qiskit.circuit.library import CZGate 
from qiskit.visualization import plot_histogram
from qiskit import transpile, assemble
from qiskit_aer import Aer 

from HouseHolder import getUnitaryHH 
from plotAll import quantumProd



def scale(num, resolution, max_value):
    return np.floor(resolution * num / max_value)

def getBars(meanPhotonNumber, datapoints, path):
    name = 'H'+str(meanPhotonNumber)+'_M_'+str(datapoints)+'.mat'
    return sio.loadmat(path+name)

def bars2bit(database, numBits):
    rows, cols = np.shape(database)

    newDataBase = np.zeros((rows*numBits,cols), dtype = np.int8) #Storage of new binvect
    #print(np.shape(newDataBase))
    resolution = 2**numBits

    newRow = np.zeros(numBits, dtype = np.int8) 
    maxVal = 1

    init = time.time()
    for i in range(cols):
        #Initial binary string
        newRow = bta.bitarray('') 

        for j in range(rows):
            
            num = np.int64(scale(database[j][i], resolution, maxVal)) # Scale the number 
            binaryNum = np.binary_repr(num, numBits) # Convert it into a binary string
            
            newRow.extend(binaryNum) # Append the entire string into de bitarray newRow
        
        
        binaryVec = np.array(newRow.tolist(), dtype = np.int8) # convert the bitarray into a np array of numBits*numBars lenght

        newDataBase[:,i] = binaryVec # assign the new vector to the column of the database
    final = time.time()

    print(final - init)
    return newDataBase

def bin2ones(database):
    rows, cols = np.shape(database)
    onesDatabase = -np.ones((rows, cols), dtype = np.int8) 
    return onesDatabase**database
    
                

def getData(meanPhotonNumber, datapoints, numBits):

    path = '/home/guerrero/Documents/UNAM/7TH-SEMESTER/ICN/References/Database/Datos_Histogramas/'
    data = getBars(meanPhotonNumber, datapoints, path)
    cohData = data['Coh']
    thData = data['Th']
    #print(np.shape(cohData))

    enCohData = bars2bit(cohData, numBits)
    enThData = bars2bit(thData, numBits)

    return enCohData, enThData

def overlap(coherent, thermal):
    np.shape(coherent)
    np.shape(thermal)
    return np.sum(np.sqrt(coherent) * np.sqrt(thermal))**2 / ((np.sum(coherent))*(np.sum(thermal)))


def normalizeRange(cohHist, thHist):
    leftTh = thHist[1][0]
    rightTh = thHist[1][len(thHist[1]) - 1]
    leftCoh = cohHist[1][0]
    rightCoh = cohHist[1][len(cohHist[1]) - 1]
    print(rightCoh - rightTh)
    fillerTh = np.zeros(int(np.sqrt(rightCoh**2 - rightTh**2)))
    fillerCoh = np.zeros(int(np.sqrt(leftTh**2 - leftCoh**2)))

    newCoh = np.concatenate((fillerCoh,cohHist[0]))
    newTh = np.concatenate((thHist, fillerTh[0]))
    return newCoh, newTh


def getTresholdInLoop(cohHist, thHist):
    #print(cohHist)
    mix = np.concatenate((cohHist[1], thHist[1]))
    min = np.min(mix)
    max = np.max(mix)
    mixArange = np.arange(min, max)
    #print(mixArange)
    
    dif = max - min 
    dif = dif / 2 
    treshold = min + dif 
    return treshold


def getFitness(mfs, coh, th, qc, nshots):

    rows, cols = np.shape(coh)
    rowsT, colsT = np.shape(th)
    projCoh = np.zeros(cols)
    projTh = np.zeros(cols)
    m = len(mfs)

    

    for i in range(cols):
        psi_i = coh[:,i]
        #projCoh[i] = len(psi_i)*quantumProd(qc, mfs, psi_i, nshots)
        cm = mfs@psi_i
        projCoh[i] = np.abs(cm/m)**2
        
        psi_i = th[:,i]
        #projTh[i] = len(psi_i)*quantumProd(qc, mfs, psi_i, nshots)
        cm = mfs@psi_i
        projTh[i] = np.abs(cm/m)**2

    n_bins = 10

    cohHist = np.histogram(projCoh, n_bins)
    thHist = np.histogram(projTh, n_bins)
    #Fitness calculation 
    #olap = overlap(thHist[0], thHist[0]) #Overlap
    olap = overlap(cohHist[0], thHist[0]) #Overlap

    treshold = getTresholdInLoop(cohHist, thHist) #accuracy
    
    mix = np.concatenate((projCoh, projTh))
    
    classification = np.zeros(len(mix))
    classification[:len(projCoh)] = 1
    
    predictions = np.where(mix >= treshold, 1, 0)

    accuracy = accuracy_score(classification, predictions)



    return olap, accuracy

def getTreshold(mfs, coh, th, qc, nshots):

    rows, cols = np.shape(coh)
    weight = mfs
    print(rows, cols)
    projCoh = np.zeros(cols)
    projTh = np.zeros(cols)
    m = len(mfs)
    
    
    for i in range(cols):
        psi_i = coh[:,i]
        #projCoh[i] = len(psi_i)*quantumProd(qc, mfs, psi_i, nshots)
        cm = mfs@psi_i
        projCoh[i] = np.abs(cm/m)**2
        
        psi_i = th[:,i]
        #projTh[i] = len(psi_i)*quantumProd(qc, mfs, psi_i, nshots)
        cm = mfs@psi_i
        projTh[i] = np.abs(cm/m)**2


    n_bins = 10
    cohHist = np.histogram(projCoh, n_bins)
    thHist = np.histogram(projTh, n_bins)

    treshold = getTresholdInLoop(cohHist, thHist)
    print(treshold)

    plt.hist(projCoh, n_bins, alpha = 0.75)
    plt.hist(projTh, n_bins, alpha = 0.75)
    plt.axvline(x = treshold, color = 'g', label = 'treshold')
    plt.legend()
    plt.title(str(mfs)+'Accuracy: '+str(overlap)  + ' treshold: ' + str(treshold))
    plt.savefig('_Accuracy_'+str(overlap)  + ' treshold: ' + str(treshold) + '.png', bbox_inches = 'tight')
    plt.close()

    
    #predCoh = np.where(projCoh >= treshold, 1, 0)
    #predTh = np.where(projTh < treshold, 1, 0)
    
    #numOfTh = np.sum(predCoh)
    #numOfCoh = np.sum(predTh)
    
    #efficiency =  ((len(th) - numOfTh) + (len(coh) - numOfCoh)) / (len(th) + len(coh))
     
    return treshold


def multipointCO(childrenMat):

    l, col = np.shape(childrenMat)
    #print(l,col)
    probArr = np.ones(l)/ l
    #print(probArr)
    ran = np.arange(0, col, 2)

    for i in ran:
        chosen = np.random.choice(np.arange(l).tolist(), 1, probArr.tolist())  
        tmp1 = childrenMat[chosen, i]
        tmp2 = childrenMat[chosen, i + 1]

        childrenMat[chosen, i] = tmp2
        childrenMat[chosen, i + 1] = tmp1

    return childrenMat

def randomMutation(population, mutVect):
    #Mut vect are the selected columns 
    cols = np.nonzero(mutVect)
    randomId = np.random.randint(0, len(population), len(cols)) # randomId are the selected entries, one per selected chromosome
    population[randomId, cols] = -1*population[randomId, cols] #switch selected mutants 

    return population

def main():
   #Parameters 
    numBits = 8

    # Data
    subs = int(input('Subset: '))
    #subs = 1000

    coh, th = getData(77, 160,numBits)
    coh, th = bin2ones(coh), bin2ones(th)
    coh, th = coh[:, :subs], th[:, :subs]

    

    ## Subset 

    cohRows, cohCols = np.shape(coh)
    thRows, thCols = np.shape(th)



    #print(coh[0:56, 1], th[0:56, 1])

    # Mixing 
    mixedStates = np.append(coh, th, axis = 1)
    mixRows, mixCols = np.shape(mixedStates)

    '''
    Quantum Circuit 
    '''

    N = 7
    Nc = 1
    qr = QuantumRegister(N - 1, 'q')
    ar = QuantumRegister(1, 'ancilla')
    cr = ClassicalRegister(2, 'c')

    qc = QuantumCircuit(qr,ar,cr)
    nshots = 100
    
    # Filler section: The residual slots need to be filled 
    fillerAllPos = np.ones(2**(qc.num_qubits - 1) - mixRows, dtype = np.int8) # Remaining slots are filled with ones 
    # [1 -1 1 -1 -1] [1 1 1 1]
    fillerHalfNeg = fillerAllPos 
    
    #Adding negatives
    fillerHalfNeg[int(len(fillerAllPos)/2):] = -1 

    # Filler in data 
    fillerMatrix = np.ones((2**(qc.num_qubits - 1) - mixRows, cohCols), dtype = np.int8)
    coh = np.concatenate((coh, fillerMatrix), axis = 0)
    th = np.concatenate((th, fillerMatrix), axis = 0)
      
   
    # Population Matrix 
    nPop = int(input('Population: '))
    population = np.random.randint(0,2,(56,nPop))
    population = bin2ones(population)

    # Filler 
    fillerWeights = np.ones((2**(qc.num_qubits - 1) - mixRows, nPop), dtype = np.int8)
    half = int(len(fillerWeights))

    population = np.concatenate((population, fillerWeights), axis = 0)


    print('Checking method')

    '''
    Genetic Optimization 
    '''
    # Parameters 
    lChrom = len(population) # The length of each chromosome 
    numEpochs = int(input('Number of epochs: ')) # Number of epochs to perform 
    bestSolution = np.zeros(numEpochs + 1) # Vector for the best solution of the epoch 
    bestChArr = np.zeros((lChrom, numEpochs + 1))
    bestOverlap = np.ones(numEpochs + 1)
    pCross = 0.75 # Probability of crossover for each chromosome, all of them have pCross prob of crossover. 
    #pMut = 0.0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001 # Probability of mutation for each chromosome 
    #numChildren = int(0.7 * nPop) #Number of children to generate in the epoch 
    #probability arrays 
    crossVect = np.random.choice([0,1], nPop, [1 - pCross, pCross]) # A vector of lenght equal to the population, stores 1 or 0 for a chromosome apt for crossover (pCross) and the not apt 
        #print('crossvect:',crossVect) 
    #print(len(crossVect))
   
    # Generate parents indices 
    parents_id = np.nonzero(crossVect) # Where the nonzero (parents) are 
    #mutation_id = np.nonzero(mutVect) # Where the nonzero (mnutants) are 
    
    _, numCrossOver = np.shape(parents_id) 
    #_, numMutants = np.shape(mutation_id)


    '''
     ******************************************** Training ****************************************************
    '''
    epoch = 1
    while epoch <= numEpochs:
        #Create fitness vector 
        fitArray = np.zeros(nPop)
        accuracyArray = np.zeros(nPop)
        #fitArrayTest = np.zeros(nPop)

        for i in range(nPop):
            fitArray[i] , accuracyArray[i]= getFitness(population[:, i] , coh, th, qc, 100) # perform the product and get the overlap, we want a minimum overlap        


        '''
        Parent selection 
        '''
        numParents = int(np.ceil(0.75 * numCrossOver / 2) * 2) # get an even number of parents
        accuracyCross = accuracyArray * crossVect # fitArray is the probability array for the selection of parents, by multiplying the entries of fitArray and crossVect we are excluding the individuals that where not selected as parents. 
        
        #Normalizing both arrays 
        accuracyCross = (1/sum(accuracyCross))*accuracyCross
        #fitArray = (1/sum(fitArray)) * fitArray
        # Now we got to choose the parents 

        parents = np.zeros(numParents)
        #mutants = np.zeros(numMutants)
        #print(np.arange(nPop))
        #print(len(fitArray))
        
        # I can't select the parents with np choice so I'll sort them by their fitness and from them choose the first numParents. An elititst strategy  
        index = np.arange(len(population[0]))
        #print(np.random.choice(index, 40, k))

        sortCO = np.sort(accuracyCross, kind = 'mergesort') # Sort by fitness 
        sortFit = np.sort(accuracyArray, kind = 'mergesort')

        '''
        ******************** Best Solution of the epoch ***********************
        '''
        bestSolution[epoch] = sortFit[-1] # We need to revert the fitness to the original value so we add 1
        np.savetxt(fname = 'bestSolution.csv', X = bestSolution, delimiter= ',')
        best_id = int(np.argwhere(accuracyArray == sortFit[len(sortFit) - 1])[:][0][0])
        
        bestOverlap[epoch] = fitArray[best_id]

        bestChromosome = population[:, best_id]
        bestChArr[:, epoch] = bestChromosome
        np.savetxt(fname = 'bestChromosome.csv', X = bestChArr, delimiter = ',')


        '''
        ***********************************************************************
        '''


        zeros_co = nPop - numCrossOver
        #print(sortCO)

        
        
        ran = np.arange(zeros_co, numParents, numParents + zeros_co)
        #print(np.size(fitArray))
        #print(np.size(index))
        
        #Choose the parents 
        parents = np.random.choice(index, numParents, replace = False, p = accuracyCross.tolist())
        #print(parents)
        '''
        Crossover 
        '''
        # Generate children 
        children = np.zeros((lChrom, numParents)) # The children of the generation 
        #print('Selected Parents:', np.shape(children))

        #print(np.shape(population))
        #print(np.shape(children))
        #print(numParents)
        #print(population[:, parents])
        children = population[:,parents]
        #print(children)
        #print(np.shape(children))

        # Now perform crossover
        newChildren = multipointCO(children)

        

        '''
        Replace 
        '''
        
        # select the weakest members 
        # We're going to replace the weakest with the children
        weak = np.arange(zeros_co, zeros_co + numParents)

        #print(np.argwhere(fitArrayCross < 0.2))
        toReplace = np.zeros(numParents, dtype = int) 
        #print(sortFit)
        #print(weak)
        for i in np.arange(0, numParents , 1):
            toReplace[i] = int(np.argwhere(accuracyArray == sortFit[i])[:][0][0])
        #print(fitArray)

        #print(toReplace)

        # Now we replace the chromosomes of the original population with the children 

        #print(population[:, toReplace])
        #oldPop = population[:, toReplace]
        population[:, toReplace] = children[:]
        #newPop = population[:, toReplace]
        #print(len(np.argwhere(oldPop == newPop)))

        '''
        Mutation 
        '''
        #pMut = 1 / (epoch * np.sqrt(lChrom))
        #mutVect = np.random.choice([0,1], nPop, [1 - pMut, pMut]) # The same but with the mutation probabilities
        # There's a problem with mutVect, altough we assign a really small probability to mutant generation
        # we are having a lot of mutants, that are transforming this GA to a random search
        # so I will create a vector with a fixed number of ones, a very low number, like 3 or 4 
        mutVect = np.zeros(nPop)
        #stdState = 0.25*numEpochs
        desiredMutants = int(0.30 * nPop) #+ numEpochs - epoch #+ np.floor(nPop/(epoch * np.sqrt(lChrom)))
        #desiredMutants = np.floor(nPop/(epoch * np.sqrt(lChrom)))

        randIndex = np.random.randint(0, nPop, desiredMutants)
        #print(randIndex)
        mutVect[randIndex] = 1 #int(np.sqrt(numEpochs - epoch))

        population = randomMutation(population, mutVect)

        print(epoch)
        epoch+=1


    minId = np.argmax(bestSolution)
    plt.close()
    plt.plot(np.arange(numEpochs + 1), bestSolution, color = 'r', label = 'Accuracy')
    plt.plot(np.arange(numEpochs + 1), bestOverlap, color = 'b', label = 'Overlap')
    plt.legend()
    plt.title('Q-Sim'+str(bestChromosome) + ' Overlap: ' + str(bestOverlap[minId]) + 'accuracy: '+ str(bestSolution[minId]))
    plt.savefig('_Accuracy_' + str(bestSolution[minId]) + '_overlap_'+ str(bestOverlap[minId]) + '.png', bbox_inches = 'tight', dpi = 300)
    plt.close()


    #Test 
    print(np.shape(bestChromosome))
    mfs = bestChArr[:, minId]
    #dataMerged = np.concatenate((coh, th), axis = 1)
    #predictions = np.zeros(len(dataMerged[0]))
    # TODO: calculate the treshold
    treshold = getTreshold(mfs,coh,th,qc, nshots)
    print(treshold)
if __name__ == "__main__":
    main()
