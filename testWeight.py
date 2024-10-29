import transforms as tr 
import plotTest as pr 
import numpy as np
import matplotlib.pyplot as plt
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
    indexCoh = np.random.randint(train + 1, cols , test)
    indexTh = np.random.randint(train + 1, cols, test)

    coh, th = coh[:, indexCoh], th[:, indexTh] #We have the test sets 
    
        
    w = np.array(w, dtype = np.int8)
      
    coh = qml.addFiller(coh, 7)
    th = qml.addFiller(th, 7)


    # Generate circuits 
    nshots = 2**7
    nshots = int(input('shots'))
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
#




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
    indexCoh = np.random.randint(train + 1, cols , test)
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
    indexCoh = np.random.randint(train + 1, cols , test)
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
        
        clasCoh.append(qml.singleProjAer(coh[:,i], w, 7, 2**9))
        clasTh.append(qml.singleProjAer(th[:,i], w, 7, 2**9))
        

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
#def testClassic(train, w, treshold):
#w = [1,-1,-1,1,1,1,1,1,1,-1,1,-1, 1,-1,1,-1,1,1, 1,1,1,-1, 1,-1,1,1, 1,1,1,-1, 1,-1,1,1, 1,1, 1,1,1,1, 1,1,1,1, -1, 1]
w = [1, -1,-1,-1, -1,1,-1,1,  1,-1,-1,1,   -1,-1,1,-1,  1,1,  
     -1,1,-1,-1,  1,-1,1,1,  1,1,-1,1,  -1,1,1,1,  1,1,  1,1,1,-1,  1,1,-1,1,  1,1,1,1,  1,1,1,1,  1,1, 1,1,1,1,1,1,1,1,1,1]
print(len(w))


accQ, mixQiskit = testSeqQiskit(1000, w, 0.24853515625)
acc, mixClass = testClassic(1000, w, 0.24853515625)
dif = []
for i in range(len(mixQiskit)):
    dif.append(np.abs(mixQiskit[i] - mixClass[i]))


plt.plot(mixQiskit, "-r", label = 'Qiskit')
plt.plot(mixClass, "-b", label = 'Classic')
plt.plot(dif, "-g", label = 'Abs Difference')
plt.legend(loc="upper left")
plt.title('Perceptrón cuántico vs clásico')
#plt.show()
plt.savefig('testWeightCuanticoVsClasico.png', bbox_inches = 'tight')
plt.close()

print('Qiskit: ', accQ)
print('Classic: ', acc)
