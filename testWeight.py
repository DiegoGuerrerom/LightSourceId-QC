import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#Quantum computing 
from qiskit import QuantumCircuit 
from qiskit import QuantumRegister 
from qiskit import ClassicalRegister 
from qiskit.circuit.library import CZGate 
from sklearn.linear_model import LinearRegression
from qiskit.visualization import plot_histogram
from qiskit import transpile, assemble
from qiskit_aer import Aer 
import qmltools as qml
import transforms as tr
from sklearn.metrics import accuracy_score




def plotHist(projCoh, projTh, treshold, diff): 
    n_bins = 15
    cohHist = np.histogram(projCoh, n_bins)
    thHist = np.histogram(projTh, n_bins)

    plt.hist(projCoh, n_bins, alpha = 0.75)
    plt.hist(projTh, n_bins, alpha = 0.75)
    plt.axvline(x = treshold, color = 'g', label = 'treshold')
    plt.legend()
    plt.title(diff  + ' treshold: ' + str(treshold))
    plt.savefig('testWeightTreshold_' + str(treshold) + '.png', bbox_inches = 'tight')
    plt.close()


def testDiagonalMethod():

    
    circuits = qml.getProjCircuits(3, data,wqe)
    
def testQiskit(train, w, treshold):
    numBits = 8
    # Load photon data 
    coh, th = qml.getData(77,160, numBits)
    rows, cols = np.shape(coh)
    # Data ecoding to -1 and 1
    coh, th = qml.bin2ones(coh), qml.bin2ones(th) # Now we have the data in ones 

    # Adding filler 
    print(np.shape(coh))
    

    # ********************* Data **********************
    
    trainPerc = 0.80
    testPerc = 1 - trainPerc
    allData =  int(train / trainPerc)
    test = int(allData*testPerc)
    print('train:', train)
    print('test:', test)



    # Making the random test set
    indexCoh = np.random.randint(train + 1, cols , test)
    indexTh = np.random.randint(train + 1, cols, test)

    coh, th = coh[:, indexCoh], th[:, indexTh] #We have the test sets 
    
        
    w = np.array(w, dtype = np.int8)
      
    coh = qml.addFiller(coh, 7)
    th = qml.addFiller(th, 7)


    # Generate circuits 
    nshots = 2**13
    #nshots = int(input('shots'))
    circuitsCoh = qml.getProjCircuits(7, coh, w)
    circuitsTh = qml.getProjCircuits(7, th, w)
    jobResultsCoh = tr.simulate_sv(circuitsCoh,  nshots)
    jobResultsTh = tr.simulate_sv(circuitsTh,  nshots)
    simuCoh = tr.getResultsFromJobs(jobResultsCoh)
    simuTh = tr.getResultsFromJobs(jobResultsTh)
    print('finished aer simulation')



    #circuitsCoh = qml.getProjCircuits(7, coh, w)
    #circuitsTh = qml.getProjCircuits(7, th, w)
    #print(circuits)
    #c = np.dot(i,w)**2 
    #p = tr.simulate(qml.getProjCircuits(coh,w), nshots) * 64

    #ran = np.arange(4,21,1)
    #ran = np.arange(1, 3, 1)
    meanErrorCoh = []
    meanErrorTh = []
    clasCoh = []
    clasTh = []
    l = len(w)
    
    mix = np.concatenate((simuCoh, simuTh))   
    classification = np.zeros(len(mix)) 
    classification[:len(simuCoh)] = 1

    
    predictions = np.where(mix >= treshold, 1, 0)

    print(np.sum(predictions))
    np.savetxt('testWeightPred.txt', predictions)
    accuracy = accuracy_score(classification, predictions)
    print(np.nonzero(predictions))

    plotHist(simuCoh, simuTh, treshold, 'qiskit')
    

    return accuracy, mix


def testRealSession(train, w, treshold):
    # ***************** Get Data *********************
    numBits = 8
    # Load photon data 
    coh, th = qml.getData(77,160, numBits)
    rows, cols = np.shape(coh)
    # Data ecoding to -1 and 1
    coh, th = qml.bin2ones(coh), qml.bin2ones(th) # Now we have the data in ones 

    # Adding filler 
    print(np.shape(coh))
    

    # ********************* Data Preparation **********************
    
    trainPerc = 0.80
    testPerc = 1 - trainPerc
    allData =  int(train / trainPerc)
    test = int(allData*testPerc)
    print('train:', train)
    print('test:', test)


    # Making the random test set
    np.random.seed(666)
    indexCoh = np.random.randint(train + 1, cols , test)
    np.random.seed(777)
    indexTh = np.random.randint(train + 1, cols, test)


    coh, th = coh[:, indexCoh], th[:, indexTh] #We have the test sets 
    
        
    w = np.array(w, dtype = np.int8)
      
    coh = qml.addFiller(coh, 7)
    th = qml.addFiller(th, 7)


    #********************** Quantum Circuits ******************** 
    nshots = 2**13
    #nshots = int(input('shots'))

    circuitsCoh = qml.getProjCircuits(7, coh, w)
    circuitsTh = qml.getProjCircuits(7, th, w)
    # ********************** Execution ************************
    job_qrealCoh = tr.realExecuteSessionMulti(circuitsCoh, nshots)
    job_qrealTh = tr.realExecuteSessionMulti(circuitsTh, nshots)
    
    realCoh = tr.getResultsFromJobs(job_qrealCoh)
    realTh = tr.getResultsFromJobs(job_qrealTh)

    print('finished Session')



    meanErrorCoh = []
    meanErrorTh = []
    clasCoh = []
    clasTh = []
    l = len(w)
    
    mix = np.concatenate((simuCoh, simuTh))   
    classification = np.zeros(len(mix)) 
    classification[:len(simuCoh)] = 1

    
    predictions = np.where(mix >= treshold, 1, 0)

    print(np.sum(predictions))
    np.savetxt('testWeightPred.txt', predictions)
    accuracy = accuracy_score(classification, predictions)
    print(np.nonzero(predictions))

    plotHist(simuCoh, simuTh, treshold, 'qiskit')
    

    return accuracy, mix



def testRealBatch(train, w, treshold):
    numBits = 8
    # Load photon data 
    coh, th = qml.getData(77,160, numBits)
    rows, cols = np.shape(coh)
    # Data ecoding to -1 and 1
    coh, th = qml.bin2ones(coh), qml.bin2ones(th) # Now we have the data in ones 

    # Adding filler 
    print(np.shape(coh))
    

    # ********************* Data **********************
    
    trainPerc = 0.80
    testPerc = 1 - trainPerc
    allData =  int(train / trainPerc)
    test = int(allData*testPerc)
    print('train:', train)
    print('test:', test)


    # Making the random test set
    np.random.seed(666)
    indexCoh = np.random.randint(train + 1, cols , test)
    np.random.seed(777)
    indexTh = np.random.randint(train + 1, cols, test)


    coh, th = coh[:, indexCoh], th[:, indexTh] #We have the test sets 
    
        
    w = np.array(w, dtype = np.int8)
      
    coh = qml.addFiller(coh, 7)
    th = qml.addFiller(th, 7)


    # Generate circuits 
    nshots = 2**13
    #nshots = int(input('shots'))

    circuitsCoh = qml.getProjCircuits(7, coh, w)
    circuitsTh = qml.getProjCircuits(7, th, w)
    
    pubCoh = tr.realExecuteBatch(circuitsCoh, nshots)
    pubTh = tr.realExecuteBatch(circuitsTh, nshots)
    
    qrealCoh = tr.getResultsBatch(pubCoh)
    qrealTh = tr.getResultsBatch(pubTh)
    np.savetxt('cohBatch.csv', qrealCoh, delimiter = ',')
    np.savetxt('ThBatch.csv', qrealTh, delimiter = ',')
    print('finished aer simulation')



    meanErrorCoh = []
    meanErrorTh = []
    clasCoh = []
    clasTh = []
    l = len(w)
    
    mix = np.concatenate((simuCoh, simuTh))   
    classification = np.zeros(len(mix)) 
    classification[:len(simuCoh)] = 1

    
    predictions = np.where(mix >= treshold, 1, 0)

    print(np.sum(predictions))
    np.savetxt('testWeightPred.csv', predictions, delimiter = ',')
    accuracy = accuracy_score(classification, predictions)
    print(np.nonzero(predictions))

    plotHist(simuCoh, simuTh, treshold, 'qiskit')
    

    return accuracy, mix




def testClassic(train, w, treshold):
    '''
    i = np.random.randint(0,2,64)
    i = (-1*np.ones_like(i))**i
    #i = np.ones(64)
    print('input:\n',i)
    w = np.random.randint(0,2,64)
    w = (-1*np.ones_like(w))**w
    #w = np.ones(64)
    print('weight:\n', w)
    #'''
    numBits = 8
    # Load photon data 
    coh, th = qml.getData(77,160, numBits)
    rows, cols = np.shape(coh)
    # Data ecoding to -1 and 1
    coh, th = qml.bin2ones(coh), qml.bin2ones(th) # Now we have the data in ones 

    # Adding filler 
    print(np.shape(coh))
    

    # ********************* Data **********************
    
    trainPerc = 0.80
    testPerc = 1 - trainPerc
    allData =  int(train / trainPerc)
    test = int(allData*testPerc)
    print('train:', train)
    print('test:', test)



    # Making the random test set
    np.random.seed(666)
    indexCoh = np.random.randint(train + 1, cols , test)
    np.random.seed(777)
    indexTh = np.random.randint(train + 1, cols, test)

    coh, th = coh[:, indexCoh], th[:, indexTh] #We have the test sets 
    
        
    w = np.array(w, dtype = np.int8)
      
    coh = qml.addFiller(coh, 7)
    th = qml.addFiller(th, 7)


    #circuitsCoh = qml.getProjCircuits(7, coh, w)
    #circuitsTh = qml.getProjCircuits(7, th, w)
    #print(circuits)
    #c = np.dot(i,w)**2 
    #p = tr.simulate(qml.getProjCircuits(coh,w), nshots) * 64

    #ran = np.arange(4,21,1)
    #ran = np.arange(1, 3, 1)
    meanErrorCoh = []
    meanErrorTh = []
    clasCoh = []
    clasTh = []
    l = len(w)
    for i in range(len(coh[0])):
        clasCoh.append(np.abs(coh[:,i]@w / l)**2)
        clasTh.append(np.abs(th[:,i]@w / l)**2)

    mix = np.concatenate((clasCoh, clasTh))   
    classification = np.zeros(len(mix))
    classification[:len(clasCoh)] = 1


    predictions = np.where(mix >= treshold, 1, 0)
    print(np.sum(predictions))
    np.savetxt('testWeightPred.txt', predictions)
    accuracy = accuracy_score(classification, predictions)
    print(np.nonzero(predictions))
    plotHist(clasCoh, clasTh, treshold, 'classic')

    return accuracy, mix




def testSeqQiskit(train, w, treshold):
    '''
    i = np.random.randint(0,2,64)
    i = (-1*np.ones_like(i))**i
    #i = np.ones(64)
    print('input:\n',i)
    w = np.random.randint(0,2,64)
    w = (-1*np.ones_like(w))**w
    #w = np.ones(64)
    print('weight:\n', w)
    #'''
    numBits = 8
    # Load photon data 
    coh, th = qml.getData(77,160, numBits)
    rows, cols = np.shape(coh)
    # Data ecoding to -1 and 1
    coh, th = qml.bin2ones(coh), qml.bin2ones(th) # Now we have the data in ones 

    # Adding filler 
    print(np.shape(coh))
    

    # ********************* Data **********************
    
    trainPerc = 0.80
    testPerc = 1 - trainPerc
    allData =  int(train / trainPerc)
    test = int(allData*testPerc)
    print('train:', train)
    print('test:', test)



    # Making the random test set
    np.random.seed(666)
    indexCoh = np.random.randint(train + 1, cols , test)
    np.random.seed(777)
    indexTh = np.random.randint(train + 1, cols, test)

    coh, th = coh[:, indexCoh], th[:, indexTh] #We have the test sets 
    
        
    w = np.array(w, dtype = np.int8)
      
    coh = qml.addFiller(coh, 7)
    th = qml.addFiller(th, 7)

    meanErrorCoh = []
    meanErrorTh = []
    clasCoh = []
    clasTh = []
    l = len(w)
    nshots = 2**13
    for i in range(len(coh[0])):
        
        clasCoh.append(qml.singleProjAer(coh[:,i], w, 7, nshots))
        clasTh.append(qml.singleProjAer(th[:,i], w, 7, nshots))
        

    mix = np.concatenate((clasCoh, clasTh))   
    classification = np.zeros(len(mix))
    classification[:len(clasCoh)] = 1


    predictions = np.where(mix >= treshold, 1, 0)
    print(np.sum(predictions))
    np.savetxt('testWeightPred.txt', predictions)
    accuracy = accuracy_score(classification, predictions)
    print(np.nonzero(predictions))
    plotHist(clasCoh, clasTh, treshold, 'classic')

    return accuracy, mix

def testFakeParallel(train, w, treshold):
    
    numBits = 8
    # Load photon data 
    coh, th = qml.getData(77,160, numBits)
    rows, cols = np.shape(coh)
    # Data ecoding to -1 and 1
    coh, th = qml.bin2ones(coh), qml.bin2ones(th) # Now we have the data in ones 

    # Adding filler 
    print(np.shape(coh))
    

    # ********************* Data **********************
    
    trainPerc = 0.80
    testPerc = 1 - trainPerc
    allData =  int(train / trainPerc)
    test = int(allData*testPerc)
    print('train:', train)
    print('test:', test)



    # Making the random test set
    indexCoh = np.random.randint(train + 1, cols , test)
    indexTh = np.random.randint(train + 1, cols, test)

    coh, th = coh[:, indexCoh], th[:, indexTh] #We have the test sets 
    
        
    w = np.array(w, dtype = np.int8)
      
    coh = qml.addFiller(coh, 7)
    th = qml.addFiller(th, 7)


    # Generate circuits 
    nshots = 2**13
    #nshots = int(input('shots'))
    circuitsCoh = qml.getProjCircuits(7, coh, w)
    circuitsTh = qml.getProjCircuits(7, th, w)

    fakeCoh = tr.simulateAerBackendParallel(circuitsCoh, nshots)
    fakeTh = tr.simulateAerBackendParallel(circuitsTh, nshots)
    print('finished aer simulation')



    #circuitsCoh = qml.getProjCircuits(7, coh, w)
    #circuitsTh = qml.getProjCircuits(7, th, w)
    #print(circuits)
    #c = np.dot(i,w)**2 
    #p = tr.simulate(qml.getProjCircuits(coh,w), nshots) * 64

    #ran = np.arange(4,21,1)
    #ran = np.arange(1, 3, 1)
    meanErrorCoh = []
    meanErrorTh = []
    clasCoh = []
    clasTh = []
    l = len(w)
    
    mix = np.concatenate((fakeCoh, fakeTh))   
    classification = np.zeros(len(mix)) 
    classification[:len(fakeCoh)] = 1

    
    predictions = np.where(mix >= treshold, 1, 0)

    print(np.sum(predictions))
    np.savetxt('testWeightPred.txt', predictions)
    accuracy = accuracy_score(classification, predictions)
    print(np.nonzero(predictions))

    plotHist(simuCoh, simuTh, treshold, 'qiskit')
    

    return accuracy, mix



def main():
    #w = [1,-1,-1,1,1,1,1,1,1,-1,1,-1, 1,-1,1,-1,1,1, 1,1,1,-1, 1,-1,1,1, 1,1,1,-1, 1,-1,1,1, 1,1, 1,1,1,1, 1,1,1,1, -1, 1]
    #w = [1, -1,-1,-1, -1,1,-1,1,  1,-1,-1,1,   -1,-1,1,-1,  1,1, -1,1,-1,-1,  1,-1,1,1,  1,1,-1,1,  -1,1,1,1,  1,1,  1,1,1,-1,  1,1,-1,1,  1,1,1,1,  1,1,1,1,  1,1, 1,1,1,1,1,1,1,1,1,1]

    w = pd.read_csv('09115bestSingleCh.csv', delimiter = ',', header = None)
    print(len(w))


    accQ, mixQiskit = testFakeParallel(1000, w, 0.24853515625)
    acc, mixClass = testClassic(1000, w, 0.24853515625)
    dif = []
    for i in range(len(mixQiskit)):
        dif.append(np.abs(mixQiskit[i] - mixClass[i]))


    plt.plot(mixQiskit, "-r", label = 'Qiskit')
    plt.plot(mixClass, "-b", label = 'Classic')
    plt.plot(dif, "-g", label = 'Abs Difference')
    plt.legend(loc="upper left")
    plt.title('Perceptrón cuántico (Qiskit) vs clásico')
    #plt.show()
    plt.savefig('testWeightCuanticoVsClasico.png', bbox_inches = 'tight')
    plt.close()

    print('Fake parallel: ', accQ)
    print('Classic: ', acc)


    #accReal, mixReal = testRealSession(100, w, 0.24853515625)
    #np.save_txt('accRealSession.csv', accReal, delimeter = ',')
    #print('Q Real: ', accReal)


main()
