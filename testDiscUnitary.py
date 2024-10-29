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


def minDiag(mat):
    rows, cols = np.shape(mat)
    diagRight = np.diag(mat)
    diagLeft = np.zeros(rows)
    for i in range(rows):
        diagLeft[i] = mat[i, rows -1 - i]   
    mix = np.concatenate((diagRight, diagLeft))
    return np.min(mix)
#Creando circuito 
N = 7
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

l = 40
#l = p
ran = np.linspace(0, p, l)
#ran = np.arange(0,p)
print(ran)
probs = np.zeros((l,l))
c_probs = np.zeros((l,l))
# Ciclo para realizar producto escalar en todos los estados
print(ran)
dataAll = []
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
        dataAll.append(data)
        w = -np.ones(m, dtype = np.int8)
        exp = bta.bitarray(np.binary_repr(kj, m)).tolist()
        w = w**exp
        proj = qml.singleProjAer(data,w,N,2**16)
        
        probs[i,j] = proj[0]     #probs[i,j] = (tr.neuron(qc,i,j)/100)
        c_probs[i,j] = np.abs(data@w / len(w))**2
'''
for i in range(int(p/2), p):
    for j in range(int(p/2), p):
        probs[i,j] = (tr.neuron_rev(qc,i,j)/100)
'''

#print(probs)

minDiag = minDiag(probs)
x, y = np.meshgrid(np.arange(probs.shape[0]), np.arange(probs.shape[1]))
z = probs.flatten()


dz = z
offset = dz + np.abs(dz.min())
fracs = offset.astype(float)/offset.max()
norm = colors.Normalize(fracs.min(), fracs.max())
color_values = cm.viridis(norm(fracs.tolist()))

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

vmin, vmax = z.min(), z.max()
ax.bar3d(x.ravel(), y.ravel(), np.zeros_like(z), 1, 1, z, shade=True, color=color_values)

plt.show()


# Heat map

def plot_heat_map(probs,title, annotations, ylab):
    fig, ax = plt.subplots()
    im = ax.imshow(probs)



    # Show all ticks and label them with the respective list entries
    ax.set_xticks(np.arange(len(y)))
    ax.set_yticks(np.arange(len(x)))


    # Loop over data dimensions and create text annotations.
    if(annotations == True):
        for i in range(len(probs)):
            for j in range(len(probs)):
                text = ax.text(j, i, np.around(probs[i, j], 3),
                            ha="center", va="center", color="w")
    #Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(ylab, rotation=-90, va="bottom")

    ax.set_title(title)
    fig.tight_layout()
    return fig


q_perceptron_fig = plot_heat_map(probs,"Simulated scalar product <Psi_i | Psi_w> minDiag: "+str(minDiag), True, 'Scalar product')

print('************ Classical Perceptron **************')


#mat_i = cp.classic_vector(n_qbits)
print("********Classic Matrix*************\n")
#print(mat_i)

#c_probs = cp.classic_perceptron(mat_i)




print('*** Classical Probabilities***')

print(c_probs)
c_perceptron_fig = plot_heat_map(c_probs, 'Classic Scalar Product <i,w>', True, 'Scalar product')


disc_matrix = tr.discrepancy(probs, c_probs)

disc_mat_fig = plot_heat_map(disc_matrix, 'Sigle Discrepancy', False, 'Discrepancy')

avg_discrepancy = tr.avg_discrepancy(disc_matrix)
print('********Average Discrepancy***********')
print(avg_discrepancy)

q_perceptron_fig
c_perceptron_fig
disc_mat_fig
plt.show()


print('************ Classical Perceptron 8 bits **************')


#mat_i = cp.classic_vector(3)
print("********Classic Matrix*************\n")
#print(mat_i)

#c_probs = cp.classic_perceptron(mat_i)




print('*** Classical Probabilities***')

print(c_probs)
np.savetxt(c_probs, 'classicalProb.csv')
c_perceptron_fig = plot_heat_map(c_probs, 'Classic Scalar Product <i,w>', True, 'Scalar product')

plt.show()


