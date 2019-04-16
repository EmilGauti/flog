# -*- coding: utf-8 -*-

# Patient-specific classifier

# Pre: Execute main.py

import ast

def fixIndex(i):
    if i < 10:
        i = '0'+str(i)
    return(str(i))

dataPath = '../data/'
nonSeizureFileNames = open(dataPath + 'nonSeizureFileNames.txt', 'r')
nonSeizureFileNames = nonSeizureFileNames.read().split('\n')

with open(dataPath +'seizureDict.txt', 'r') as f:
    s = f.read()
    seizureDict = ast.literal_eval(s)

for testPatient in range(1,nrPatients+1):
    #gets index of seizures that belong to testPatient (+nrSeizures since seizure chunks are first in allData)
    prefix = 'chb'+fixIndex(testPatient)
    nonSeizIndices = [i+nrSeizures for i, s in enumerate(nonSeizureFileNames) if s.startswith(prefix)]
    testPatientKeys = [i for i in seizureDict.keys() if i.startswith(prefix)]
    k = 0
    seizIndices = []
    for key in seizureDict:    
        for i in range(len(seizureDict[key])):
            if key in testPatientKeys:
                seizIndices.append(k)
            k = k+1

    print("Patient ", testPatient, "n_seizures=",len(seizIndices), "n_nonseizures=",len(nonSeizIndices))
    
    X_test = np.concatenate((X[seizIndices,:],X[nonSeizIndices,:]),axis = 0)
    y_test = np.concatenate((y[seizIndices],y[nonSeizIndices]))
    X_train = np.delete(X, nonSeizIndices, axis=0)
    X_train = np.delete(X, seizIndices, axis=0)
    y_train = np.delete(y,nonSeizIndices)
    y_train = np.delete(y,seizIndices)
    # THINK: Collect statistics on seizure/nonseizure
    print("Train:", X_train.shape, "Test:", X_test.shape)

    # Classify individual patient
    # Insert code here ...
