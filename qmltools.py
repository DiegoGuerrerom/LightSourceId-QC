import numpy as np
import matplotlib.pyplot as plt 
import scipy.io as sio 
import bitarray as bta 
import bitarray.util as butil 
import transforms as tr 
import time 
from qiskit.quantum_info import Statevector


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
from qiskit.circuit.library import MCXGate, UnitaryGate


from HouseHolder import getUnitaryHH, getUiSimple
from plotAll import quantumRealProd, quantumProd
import transforms as tr 


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

    path = '/home/guerrero/Documents/UNAM/PerceprontV1/database/'
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

    mix = np.concatenate((cohHist[1], thHist[1]))
    min = np.min(mix)
    max = np.max(mix)
    mixArange = np.arange(min, max)
    treshold = np.median(mixArange)
    return treshold


def getAllFitness(mfs, coh, th, qc, treshold, nshots):
    
    w = np.array(mfs, dtype = np.int8)
    rows, cols = np.shape(coh)
    rowsT, colsT = np.shape(th)
    
    # Classical projections 
    projCoh = np.zeros(cols)
    projTh = np.zeros(cols)

    #Qiskit simulated projections 
    projCohSim = np.zeros(cols)
    projThSim = np.zeros(cols)
    
    #Executed in IBMQ processors
    projCohQuant = np.zeros(cols)
    projThQuant = np.zeros(cols)
    
    for i in range(cols):

        # Coherent projections
        psi_i = np.array(coh[:,i], dtype = np.int8)
        print('i:', psi_i)
        print(np.dot(mfs, psi_i))
        print(mfs)
        projCoh[i] = np.dot(mfs,psi_i)
        projCohSim[i]= len(psi_i)*quantumProd(qc, mfs, psi_i, nshots)
        #projCohQuant[i] = len(psi_i)*quantumRealProd(qc, mfs, psi_i)
        
        # Thermal projections 
        psi_i = th[:,i]
        projTh[i] = np.dot(mfs, psi_i)
        projThSim[i]= len(psi_i)*quantumProd(qc, mfs, psi_i, nshots)
        #projThQuant[i] = len(psi_i)*quantumRealProd(qc, mfs, psi_i)     
    print('Cls:\n',projCoh)
    print('Sim:\n',projCohSim)
    projDiffSim_Class = np.abs(projCohSim - projCoh)
    projDiffReal_Class = np.abs(projCohQuant - projCoh)
    projDiffReal_Sim = np.abs(projCohQuant - projCohSim)

    plt.plot(np.arange(len(projCoh)), projDiffSim_Class)
    #plt.plot(np.arange(len(projCoh)), projDiffReal_Class)
    #plt.plot(np.arange(len(projCoh)), projDiffSim_Class)
    plt.savefig('differences.png')
    plt.close()
    n_bins = 10
    
    # Histogram and treshold
    cohHist = np.histogram(projCoh, n_bins)
    thHist = np.histogram(projTh, n_bins)
    #Fitness calculation 
    #olap = overlap(thHist[0], thHist[0]) #Overlap
    olap = overlap(cohHist[0], thHist[0]) #Overlap

    #treshold = getTresholdInLoop(cohHist, thHist) #accuracy
    
    # Get accuracy of each method (class, sim, real)

    mixClass = np.concatenate((projCoh, projTh))
    mixQsim = np.concatenate((projCohSim, projThSim))
    mixQreal = np.concatenate((projCohQuant, projThQuant))

    classification = np.zeros(len(mixClass))
    classification[:len(projCoh)] = 1
    
    predictionsClss= np.where(mixClass >= treshold, 1, 0)
    predictionsQsim = np.where(mixQsim >= treshold, 1, 0)
    predictionsQreal = np.where(mixQreal >= treshold, 1, 0)
    print('Classic:',predictionsClss)
    print('Simulated:',predictionsQsim)
    accuracy = (accuracy_score(classification, predictionsClss),
                accuracy_score(classification, predictionsQsim),
                accuracy_score(classification, predictionsQreal))



    return accuracy


def getFitness(mfs, coh, th, qc, treshold, nshots):
    
    rows, cols = np.shape(coh)
    rowsT, colsT = np.shape(th)
    
    # Classical projections 
    projCoh = np.zeros(cols)
    projTh = np.zeros(cols)

    #Qiskit simulated projections 
    projCohSim = np.zeros(cols)
    projThSim = np.zeros(cols)
    
    #Ejecuted in IBMQ processors
    projCohQuant = np.zeros(cols)
    projThQuant = np.zeros(cols)

    for i in range(cols):
      psi_i = coh[:,i]
      projCohQuant[i] = len(psi_i)*quantumRealProd(qc, mfs, psi_i)
      #projCoh[i] = mfs@psi_i

      psi_i = th[:,i]
      projThQuant[i] = len(psi_i)*quantumRealProd(qc, mfs, psi_i)
      #projTh[i] = mfs@psi_i

    n_bins = 10

    cohHist = np.histogram(projCoh, n_bins)
    thHist = np.histogram(projTh, n_bins)
    #Fitness calculation 
    #olap = overlap(thHist[0], thHist[0]) #Overlap
    olap = overlap(cohHist[0], thHist[0]) #Overlap

    #treshold = getTresholdInLoop(cohHist, thHist) #accuracy
    
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
    
    
    for i in range(cols):
        psi_i = coh[:,i]
        print(np.shape(weight))
        print(np.shape(psi_i))
        projCoh[i] = len(psi_i)*quantumProd(qc, mfs, psi_i, nshots)
        #projCoh[i] = weight@psi_i

        psi_i = th[:,i]
        projTh[i] = len(psi_i)*quantumProd(qc, mfs, psi_i, nshots)
        #projTh[i] = weight@psi_i

    n_bins = 10
    cohHist = np.histogram(projCoh, n_bins)
    thHist = np.histogram(projTh, n_bins)

    mix = np.concatenate((cohHist[1], thHist[1]))
    min = np.min(mix)
    max = np.max(mix)
    mixArange = np.arange(min, max)
    treshold = np.median(mixArange)
    print(treshold)

    plt.hist(projCoh, n_bins, alpha = 0.75)
    plt.hist(projTh, n_bins, alpha = 0.75)
    plt.axvline(x = treshold, color = 'g', label = 'treshold')
    plt.legend()
    plt.title(str(mfs)+'Accuracy: '+str(overlap)  + ' treshold: ' + str(treshold))
    plt.savefig('_Accuracy_'+str(overlap)  + ' treshold: ' + str(treshold) + '.png', bbox_inches = 'tight')
    plt.close()

    
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





def quantumProd(qc, mfs, psi_i, nshots):
        
    ''' Quantum Circuit: Preparation '''
    qc.clear()
    N = qc.num_qubits
    psi_plus = np.ones(2**(qc.num_qubits - 1))

    # Hadammard
    qc.h(np.arange(N - 1).tolist())
    
    # Creamos las transformaciones unitarias 
    
    # Ui
    #uiMat = getUnitaryHH(psi_plus, psi_i)
    # Testing the new simple method 
    uiMat = getUiSimple(psi_i)
    #print(uiMat)
    # Ancilla
    uiMatAnc = uiMat
    #uiMatAnc = np.kron(np.eye(2), uiMat)
    
    # Add it to the circuit
    ui = UnitaryGate(uiMatAnc)
    target = np.arange(N - 1).tolist()
    qc.append(ui, target)
    #print('StVec for psi_i=', psi_i)
    print('Numerical Vector:', psi_i)
    #print('State Vector:', Statevector(qc))
    st = Statevector(qc)
    # Check transforms 
    checkUI(st, psi_i, 64)
    
    
    # Uw 
    
    #uwMat = getUnitaryHH(mfs, psi_plus)
    # Testing the new get Uw method
    uwMat = getUiSimple(mfs)
    # Ancilla
    #uwMatAnc = np.kron(np.eye(2), uwMat)
    uwMatAnc = uwMat
    # Add it to the circuit
    uw = UnitaryGate(uwMatAnc)
    
    
    qc.append(uw, target)

    
    # Contract the state to | 0 > tensor N 
    #qc.h(np.arange(N - 1).tolist())
    # Now flip it to | 1 > tensor N 
    #qc.x(np.arange(N - 1).tolist())

    # Entangle with the ancilla
    qc.mcx(np.arange(N - 1).tolist(), N - 1)
    
    qc.measure(N - 1,0)
    print(qc)

    '''Simulating the circuit'''
    q_prod_test = np.sqrt(tr.simulate([qc], nshots)['1'] / nshots)

    return q_prod_test

def addFiller(data, num_qubits):
    # Filler section: The residual slots need to be filled 
    rows, cols = np.shape(data)
    
    fillerAllPos = np.ones(2**(num_qubits - 1) - rows, dtype = np.int8) # Remaining slots are filled with ones 
    # [1 -1 1 -1 -1] [1 1 1 1]
    fillerHalfNeg = fillerAllPos 

    #Adding negatives
    fillerHalfNeg[int(len(fillerAllPos)/2):] = -1 

    # Filler in data 
    fillerMatrix = np.ones((2**(num_qubits - 1) - rows, cols), dtype = np.int8)
    data = np.concatenate((data, fillerMatrix), axis = 0)
    
    return data


def getProjCircuits(num_qubits, data, w):

    rows, cols = np.shape(data)

    '''
    Quantum Circuit 
    '''

    N = num_qubits
    Nc = 1
    qr = QuantumRegister(N - 1, 'q')
    ar = QuantumRegister(1, 'ancilla')
    cr = ClassicalRegister(1, 'c')

    qc = QuantumCircuit(qr,ar,cr)
    #nshots = 100

    circuits = []
    psi_plus = np.ones(2**(N - 1))
    # Almost all circuits act on all qubits except for the Ancilla
    # so I'll define a target array to indicate those target qubits 
    target = np.arange(N - 1).tolist()
    
    for i in range(cols):
        qc.clear()

        # Initialize in | + >
        qc.h(target)

        # Unitary transforms 
        uiMat = getUiSimple(data[:,i])
        uwMat = getUiSimple(w) 
        print('Gate shape', np.shape(uiMat))
        #print(uiMat)
        #print(uwMat)
        np.savetxt('ui.txt', uiMat)
        # Transform them into unitary gates 
        ui = UnitaryGate(uiMat)
        uw = UnitaryGate(uwMat)

        # Add them to the circuit (build the perceptron) 
        qc.append(ui, target)
        qc.append(uw, target)
        qc.h(target)
        qc.x(target)

        # Entangle the qubits with the ancilla qubit 
        qc.mcx(target, N - 1)
        # Measure the ancilla qubit 
        qc.measure(N - 1, 0)
        print(qc)
        circuits.append(qc)
    return circuits
    # Now we create the 

def singleProjAer(i,w, num_qubits, n_shots):
    '''
    Quantum Circuit 
    '''

    N = num_qubits
    Nc = 1
    qr = QuantumRegister(N - 1, 'q')
    ar = QuantumRegister(1, 'ancilla')
    cr = ClassicalRegister(1, 'c')

    qc = QuantumCircuit(qr,ar,cr)
    
    psi_plus = np.ones(2**(N - 1))
    # Almost all circuits act on all qubits except for the Ancilla
    # so I'll define a target array to indicate those target qubits 
    target = np.arange(N - 1).tolist()
    
    # Initialize in | + >
    qc.h(target)

    # Unitary transforms 
    uiMat = getUiSimple(i)
    uwMat = getUiSimple(w) 
    print('Gate shape', np.shape(uiMat))
    #print(uiMat)
    #print(uwMat)
    np.savetxt('ui.txt', uiMat)
    # Transform them into unitary gates 
    ui = UnitaryGate(uiMat)
    uw = UnitaryGate(uwMat)

    # Add them to the circuit (build the perceptron) 
    qc.append(ui, target)
    qc.append(uw, target)
    qc.h(target)
    qc.x(target)

    # Entangle the qubits with the ancilla qubit 
    qc.mcx(target, N - 1)
    # Measure the ancilla qubit 
    qc.measure(N - 1, 0)
    print(qc)
    

    simJob = tr.simulate_sv([qc], n_shots)
    simResult = tr.getResultsFromJobs(simJob)
    return simResult



def singleProjFake(i,w,num_qubits, n_shots):
    '''
    Quantum Circuit 
    '''

    N = num_qubits
    Nc = 1
    qr = QuantumRegister(N - 1, 'q')
    ar = QuantumRegister(1, 'ancilla')
    cr = ClassicalRegister(1, 'c')

    qc = QuantumCircuit(qr,ar,cr)
    
    psi_plus = np.ones(2**(N - 1))
    # Almost all circuits act on all qubits except for the Ancilla
    # so I'll define a target array to indicate those target qubits 
    target = np.arange(N - 1).tolist()
    
    # Initialize in | + >
    qc.h(target)
    stateVec = Statevector(qc)
    print(stateVec)


    # Unitary transforms 
    uiMat = getUiSimple(i)
    uwMat = getUiSimple(w) 
    print('Gate shape', np.shape(uiMat))
    #print(uiMat)
    #print(uwMat)
    np.savetxt('ui.txt', uiMat)
    # Transform them into unitary gates 
    ui = UnitaryGate(uiMat)
    uw = UnitaryGate(uwMat)

    # Add them to the circuit (build the perceptron) 
    qc.append(ui, target)
    qc.append(uw, target)
    qc.h(target)
    qc.x(target)

    # Entangle the qubits with the ancilla qubit 
    qc.mcx(target, N - 1)
    # Measure the ancilla qubit 
    qc.measure(N - 1, 0)
    print(qc)
    

    simJob = tr.simulate([qc], n_shots)
    simResult = tr.getResultsFromJobs(simJob)
    return simResult


def singleProjReal(i,w,num_qubits, n_shots):
    '''
    Quantum Circuit 
    '''

    N = num_qubits
    Nc = 1
    qr = QuantumRegister(N - 1, 'q')
    ar = QuantumRegister(1, 'ancilla')
    cr = ClassicalRegister(1, 'c')

    qc = QuantumCircuit(qr,ar,cr)
    
    psi_plus = np.ones(2**(N - 1))
    # Almost all circuits act on all qubits except for the Ancilla
    # so I'll define a target array to indicate those target qubits 
    target = np.arange(N - 1).tolist()
    
    # Initialize in | + >
    qc.h(target)

    # Unitary transforms 
    uiMat = getUiSimple(i)
    uwMat = getUiSimple(w) 
    print('Gate shape', np.shape(uiMat))
    #print(uiMat)
    #print(uwMat)
    np.savetxt('ui.txt', uiMat)
    # Transform them into unitary gates 
    ui = UnitaryGate(uiMat)
    uw = UnitaryGate(uwMat)

    # Add them to the circuit (build the perceptron) 
    qc.append(ui, target)
    qc.append(uw, target)
    qc.h(target)
    qc.x(target)

    # Entangle the qubits with the ancilla qubit 
    qc.mcx(target, N - 1)
    # Measure the ancilla qubit 
    qc.measure(N - 1, 0)
    print(qc)
    

    simJob = tr.realExecute(qc)
    simResult = tr.getResultsFromJobs(simJob)
    return simResult


