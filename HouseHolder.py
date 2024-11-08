import IPython as ipt
import numpy as np
from qiskit.visualization import array_to_latex
from scipy.linalg import pinv
def getUnitaryHH(x, y):
    # Given vectors
    #x = np.array([1, 1, 1, 1, 1, 1, 1])
    #y = np.array([-1, 1, -1, -1, -1, -1, -1])

    # Step 1: Normalize the vectors
    x_norm = x / np.linalg.norm(x)
    y_norm = y / np.linalg.norm(y)

    # Step 2: Construct the Householder matrix
    v = x_norm - y_norm
    v = v / np.linalg.norm(v)  # Normalize v
    H = np.eye(len(x)) - 2 * np.outer(v, v)

    # Verify H is unitary: H.T @ H should be the identity matrix
    #print("H.T @ H is identity:")
    #print(np.allclose(H.T @ H, np.eye(len(x))))

    # Step 3: Scale the Householder matrix
    scale_factor = np.linalg.norm(y) / np.linalg.norm(x)
    A = H * scale_factor

    # Verify A @ x = y
    #print("A @ x equals y:")
    #print(np.allclose(A @ x, y))
    #print(A@x)

    #print("Unitary matrix A:")
    #print(A)


    #ltx = array_to_latex(np.array(A), prefix='H=')
    #ipt.display.display(ltx)
    

    A = np.eye(len(x))

    for i in range(len(x)):
        A[i,i] = y[i]

    return A


def getUiSimple(desiredVec):
    A = np.eye(len(desiredVec))

    for i in range(len(desiredVec)):
        A[i,i] = desiredVec[i]

    return A




