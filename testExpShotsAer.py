#Qiskit libraries
from qiskit import QuantumCircuit
from qiskit import QuantumRegister
from qiskit import ClassicalRegister
from qiskit.circuit.library import CZGate
from qiskit.primitives import Sampler
from qiskit.visualization import plot_histogram
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_bloch_multivector
from qiskit import QuantumCircuit, transpile, assemble
from qiskit_aer import Aer
import matplotlib.colors as colors
import matplotlib.cm as cm
from qiskit.circuit.library import MCXGate
import bitarray as bta
# Perceptron libraries 
import classicalProduct as cp
import transforms as tr

#visualization libraries 
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from colorspacious import cspace_converter
import numpy as np
from matplotlib.colors import LightSource, LinearSegmentedColormap
import qmltools as qml 
def getRegression(x, errors):
    from scipy import stats

    # Example error vector
    #errors = meanErrorTh  # replace with your actual error vector

    # Independent variable (indices)
    #x = ran

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, errors)

    # Regression line values
    regression_line = slope * x + intercept
    return x, regression_line, slope


def minDiag(mat):
    rows, cols = np.shape(mat)
    diagRight = np.diag(mat)
    diagLeft = np.zeros(rows)
    for i in range(rows):
        diagLeft[i] = mat[i, rows -1 - i]   
    mix = np.concatenate((diagRight, diagLeft))
    return np.min(mix)

def main():

    #Creando circuito 
    N = 3
    Nc = 1
    qr = QuantumRegister(N - 1, 'q')
    ar = QuantumRegister(1, 'ancilla')
    cr = ClassicalRegister(2, 'c')

    qc = QuantumCircuit(qr,ar,cr)

    print(qc.num_qubits)

    # Creando matriz de probabilidades
    n_qbits = N - 1
    m = 2**n_qbits
    p = 2**m

    #probs = np.zeros((p, p))
    print('******* Quantum Simulated Probabilities*********\n')
    #print(probs)

    #l = 10
    l = p
    #ran = np.linspace(0, p, l)
    ran = np.arange(0,p)
    print(ran)
    probs = np.zeros((l,l))
    c_probs = np.zeros((l,l))
    # Ciclo para realizar producto escalar en todos los estados
    print(ran)
    nshots_exp = np.arange(8,16)
    nshots = 2**nshots_exp
    print(nshots)
    avgDisc = []
    for shots in nshots:
        for i in range(l):
            for j in range(l):
                #unit = qml.singleProjFake(data,w, 3 , 1024)
                ki = np.int64(ran[i])
                kj = np.int64(ran[j])
                data = -np.ones(m, dtype = np.int8)
                print(len(data))
                exp = bta.bitarray(np.binary_repr(ki, m)).tolist()
                data = data**exp
                #print(data)

                w = -np.ones(m, dtype = np.int8)
                exp = bta.bitarray(np.binary_repr(kj, m)).tolist()
                w = w**exp
                proj = qml.singleProjAer(data,w,N,shots)
                
                probs[i,j] = proj[0]     #probs[i,j] = (tr.neuron(qc,i,j)/100)
                c_probs[i,j] = np.abs(data@w / len(w))**2
        '''
        for i in range(int(p/2), p):
        for j in range(int(p/2), p):
            probs[i,j] = (tr.neuron_rev(qc,i,j)/100)
        '''

        #print(probs)

        #minDiag = minDiag(probs)

        disc_matrix = tr.discrepancy(probs, c_probs)

        avg_discrepancy = tr.avg_discrepancy(disc_matrix)
        print('********Average Discrepancy***********')
        print(avg_discrepancy)
        avgDisc.append(avg_discrepancy)



    plt.close()
    x, regression_line, slope = getRegression(nshots, avgDisc)
    
    plt.plot(x, avgDisc)
    plt.plot(x, avgDisc, 'o', label='Mean difference')
    plt.plot(x, regression_line, 'r', label='Regression slope='+str(slope))

    plt.xlabel('Number of shots')
    plt.ylabel('Error')
    plt.title('Mean difference between classical and qiskit simulated perceptron increasing exp the nshots')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
