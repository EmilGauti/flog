import numpy as np
from scipy import signal

# Hjorth parameters (time domain)
# https://en.wikipedia.org/wiki/Hjorth_parameters
def hjorth_mobility(x):
    num = np.var(np.diff(x))
    den = np.var(x)
    if den > 0:
        return np.sqrt(num / den)
    else:
        return 0.0

def hjorth_parameters(x):
    activity=np.var(x)
    mobility=hjorth_mobility(x)
    if mobility > 0:
        complexity=hjorth_mobility(np.diff(x)) / mobility
    else:
        complexity=0.0
    return np.array([activity, mobility, complexity])

#Calculates power spectral density for an eeg segment
#inputs: 
#signalMat: signal matrix for chunk
#fs: sampling density
#n: number of channels
#outputs: 
#f: frequency
#Pwelch: power spectral density calculated by Welch's method
def psd(signalMat,fs,n,fRange):
    Pwelch = np.zeros((n,fRange))
    for i in range(n):
        F,Pwelch[i,:] = signal.welch(signalMat[i,:],fs,scaling = 'spectrum')
    return(F,Pwelch)

# Absolute band power
# Combined power in M frequency bands
def absolute_power(f, PSD,M,l,h):
    length = (h-l)/M
    power = []
    k = l
    for i in range(M):
        power.append(sum(PSD[np.where((f > k) & (f <= k+length))]))
        k +=length
    return(power)
    
#Relative power of delta, theta, alpha 
#and beta waves for a single channel
def relative_power(f,PSD,M,l,h):
    absPow = absolute_power(f,PSD,M,l,h)
    tot = sum(PSD)
    if tot > 0.0:
        return(absPow/tot)
    else:
        return 0.0

# Calculate relative band power for the whole data set
# THINK: Might want to do the same for absolute power
def relative_power_all(allData, nrSeizures, nrChannels, fRange, fs, M, fLowerLimit, fUpperLimit):
    dataPSD = np.zeros((nrSeizures*2,nrChannels,fRange))
    for i in range(nrSeizures*2):
        [F,dataPSD[i,:,:]] = psd(allData[i,:,:],fs,nrChannels,fRange)

    dataRelPower = np.zeros((nrSeizures*2,nrChannels,M))
    for i in range(nrSeizures*2):
        for j in range(nrChannels):
            dataRelPower[i,j,:] = relative_power(F,dataPSD[i,j,:],M,fLowerLimit,fUpperLimit)
    return(dataRelPower)
    
def create_data_matrix(allData, flatten, nrSeizures, nrChannels, fRange, fs, M, fLowerLimit, fUpperLimit):
    dataRelPower = relative_power_all(allData, nrSeizures, nrChannels, fRange, fs, M, fLowerLimit, fUpperLimit)

    if flatten:
        dataRelPowerFlat = np.zeros((nrSeizures*2,M*nrChannels))
        for i in range(nrSeizures*2):
            dataRelPowerFlat[i,:] = dataRelPower[i,:,:].flatten()
        X = dataRelPowerFlat
        y = np.concatenate((np.repeat(1,nrSeizures),np.repeat(0,nrSeizures)))
    else:
        # (channels*features matrix, e.g. for use in convolutional nets)
        X = np.expand_dims(dataRelPower,axis = 3)
        y = np.concatenate((np.repeat(1,nrSeizures),np.repeat(0,nrSeizures)))
    return(X,y)
