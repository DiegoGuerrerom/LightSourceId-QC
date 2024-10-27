'''
  ___   ____    _      _____          _       _             
 / _ \ / ___|  / \    |_   _| __ __ _(_)_ __ (_)_ __   __ _ 
| | | | |  _  / _ \     | || '__/ _` | | '_ \| | '_ \ / _` |
| |_| | |_| |/ ___ \    | || | | (_| | | | | | | | | | (_| |
 \__\_\\____/_/   \_\   |_||_|  \__,_|_|_| |_|_|_| |_|\__, |
                                                      |___/ 
Author: J.Diego Guerrero-Morales
Contact: diegoguerrerom174@gmail.com
'''

import numpy as np
import bitarray as bta 
import transforms as tr 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split 
import plotTest as qml
import matplotlib.pyplot as plt



'''

█▀▄ ▄▀█ ▀█▀ ▄▀█
█▄▀ █▀█ ░█░ █▀█

'''
def fitness(chromosome, valData, classificationVal):
    #Returns the fitness of the chromosome (state), fitness -> efficiency
    predictions_val = np.zeros_like(classificationVal)

    for i in range(len(valData)):
        proj_i = chromosome@valData[i]
        predictions_val[i] = 1*( proj_i >= treshold)
    #Calculate Confussion Matrix
    cfMatVal = confusion_matrix(classificationVal, predictions_val)
    efficiency = (np.trace(cfMatVal) / len(predictions_val))*100

    return efficiency

def main(): 
    numBits = 8
    numBars = 7
    meanPhNum = 77
    datapoints = 160 

    numBitsVec = np.arange(1,numBits + 1, 1)
    numBarsVec = np.arange(3, numBars + 1, 1)
    meanPhNumVec = np.array([4, 53, 67, 77, 735])
    datapointsVec = np.arange(30, 170, 10)
    norm = numBits*numBars

    print('numbits * numBars =', numBits, '*', numBars, '=', numBits*numBars)

    binData, decData = qml.getData(numBits, meanPhNum, datapoints)
    numObs = len(decData['Coh']) + len(decData['Th']) 


    # Trained weightVec 
    binMFS = np.array([1,-1,-1,1,  -1,1,1,1,  1,-1,1,-1,  1,-1,1,-1,  1,-1,-1,1,  -1,-1,1,1,  -1,1,1,-1,  1,-1,1,-1,  1,-1,1,1,  1,1,1,1,  -1,-1,1,1,  -1,1,1,1,  -1,-1,1,-1,  1,-1,1,1], dtype = np.int8)


    # Adjust to have matrices
    x = qml.toMatrix(qml.zero2minus(binData['Coh']), numBits, numBars).T
    y = qml.toMatrix(qml.zero2minus(binData['Th']), numBits, numBars).T

    train_percentage = 0.70
    testSize = 1 - train_percentage 

    # Train and test subset
    cohTrain, cohTest, thTrain, thTest =  train_test_split(x, y, test_size = testSize, random_state = 42)

    # Validation test 

    valCohSize = int(len(cohTest)/2)
    cohVal = cohTest[valCohSize:]
    cohTest = np.delete(cohTest, np.arange(valCohSize), axis = 0)

    valThSize = int(len(thTest)/2)
    thVal = thTest[valThSize:]
    thTest = np.delete(thTest, np.arange(valThSize), axis = 0)

    # Mixing the states

    '''
    Now we mix all the states 
    '''
    #Train
    allDataTrain = np.concatenate((cohTrain, thTrain), axis = 0)

    #Test
    allDataTest = np.concatenate((cohTest, thTest), axis = 0)

    #Valitation
    allDataVal = np.concatenate((cohVal, thVal), axis = 0)

    # The data should look like a step function
    classificationTrain = np.zeros(len(allDataTrain))
    classificationTest = np.zeros(len(allDataTest))

    classificationVal = np.zeros(len(allDataVal))

    # 0 is thermal, 1 is coherent so 
    classificationTrain[0:len(cohTrain)] = 1
    classificationTest[0:len(cohTest)] = 1
    classificationVal[0:len(cohVal)] = 1
    
    #Shuffle the data

    # Create idex
    permuted_id_train = np.random.permutation(len(allDataTrain))
    permuted_id_test = np.random.permutation(len(allDataTest))
    permuted_id_val = np.random.permutation(len(allDataVal))

    # Do permutation
    allDataTrain = allDataTrain[permuted_id_train]
    allDataTest = allDataTest[permuted_id_test]
    allDataVal = allDataVal[permuted_id_val]

    # Make the canges in classification
    classificationTrain = classificationTrain[permuted_id_train]
    classificationTest = classificationTest[permuted_id_test]
    classificationVal = classificationVal[permuted_id_val]


if __name__ == "__main__":
    main()

