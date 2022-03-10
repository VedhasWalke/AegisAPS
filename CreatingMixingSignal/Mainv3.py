from audioop import avg
import numpy as np
from scipy.io import wavfile
from os import listdir
from sox import file_info
import math
import time
import matplotlib.pyplot as plt
import csv
import wave
from datetime import date

freqPrecision = 0.01
maxFreq = 20000

avgAmplitudesByFreq = [[a*freqPrecision,0] for a in range(0,int(maxFreq/freqPrecision), 1)]
currentAmplitudesByFreq = [[a*freqPrecision,0] for a in range(0,int(maxFreq/freqPrecision), 1)]
numberOfSongsAdded = 0 # number of songs currently in the averages of avgAmplitudesByFreq

clock = time.time()
for wavFileName in listdir('../resources/'):
    
    wavFilePath = '../resources/'+wavFileName
    if(file_info.duration(wavFilePath) <= 1/freqPrecision): # discarding songs less than 1/freqprecision seconds long
        continue
    sr, data = wavfile.read(wavFilePath)

    if(file_info.channels(wavFilePath) == 2): # stereo
        data1 = [a[1] for a in data]
        data = [a[0] for a in data]

        for f in currentAmplitudesByFreq: # resetting currentAmplitudesByFreq
            f[1] = 0

        X = np.fft.fft(data1)
        N = len(X)
        n = np.arange(N)
        T = N/sr
        freqs = n/T
        freqs = [x for x in freqs if x < maxFreq]
        
        current_freq = -1
        AmplitudeSum = 0
        for i in range(0,len(freqs)):
            
            if(math.floor(freqs[i]/freqPrecision)*freqPrecision == current_freq):
                continue
            else:
                current_freq = math.floor(freqs[i]/freqPrecision)*freqPrecision
            AmplitudeSum+=abs(X[i])
        AmplitudeSum /= 1000

        current_freq = -1
        for i in range(0,len(freqs)):
            if(math.floor(freqs[i]/freqPrecision)*freqPrecision == current_freq):
                continue
            else:
                current_freq = math.floor(freqs[i]/freqPrecision)*freqPrecision
            
            avgAmplitudesByFreq[int(current_freq/freqPrecision)][1] = (avgAmplitudesByFreq[int(current_freq/freqPrecision)][1]*(numberOfSongsAdded)+abs(X[i])/AmplitudeSum)/(numberOfSongsAdded+1) # abs to get magnitude from complex number result of fft, divide by AmplitudeSum to normalize
            currentAmplitudesByFreq[int(current_freq/freqPrecision)][1] = abs(X[i])/AmplitudeSum
        numberOfSongsAdded += 1
        print("Phase Complete ("+str(time.time()-clock)+")")

        # Plotting
        plt.figure(figsize = (12, 6))
        plt.subplot(121)
        plt.plot(freqs[::1], [abs(a)/AmplitudeSum for a in X[:len(freqs):1]])
        #plt.plot([a[0] for a in currentAmplitudesByFreq[::1]], [a[1] for a in currentAmplitudesByFreq[::1]])
        plt.xlabel('Freq (Hz)')
        plt.ylabel('FFT Amplitude |X(freq)| Current')
        plt.subplot(122)
        plt.plot([a[0] for a in avgAmplitudesByFreq[::1]], [a[1] for a in avgAmplitudesByFreq[::1]])
        plt.xlabel('Freq (Hz)')
        plt.ylabel('FFT Amplitude |X(freq)| Avg')
        plt.tight_layout()
        plt.show()
        #

        clock = time.time()
    
    for f in currentAmplitudesByFreq: # resetting currentAmplitudesByFreq
        f[1] = 0

    X = np.fft.fft(data1)
    N = len(X)
    n = np.arange(N)
    T = N/sr
    freqs = n/T
    freqs = [x for x in freqs if x < maxFreq]
    
    current_freq = -1
    AmplitudeSum = 0
    for i in range(0,len(freqs)):
        
        if(math.floor(freqs[i]/freqPrecision)*freqPrecision == current_freq):
            continue
        else:
            current_freq = math.floor(freqs[i]/freqPrecision)*freqPrecision
        AmplitudeSum+=abs(X[i])
    AmplitudeSum /= 1000

    current_freq = -1
    for i in range(0,len(freqs)):
        if(math.floor(freqs[i]/freqPrecision)*freqPrecision == current_freq):
            continue
        else:
            current_freq = math.floor(freqs[i]/freqPrecision)*freqPrecision
        
        avgAmplitudesByFreq[int(current_freq/freqPrecision)][1] = (avgAmplitudesByFreq[int(current_freq/freqPrecision)][1]*(numberOfSongsAdded)+abs(X[i])/AmplitudeSum)/(numberOfSongsAdded+1) # abs to get magnitude from complex number result of fft, divide by AmplitudeSum to normalize
        currentAmplitudesByFreq[int(current_freq/freqPrecision)][1] = abs(X[i])/AmplitudeSum
    numberOfSongsAdded += 1
    print("Phase Complete ("+str(time.time()-clock)+")")

    # Plotting
    plt.figure(figsize = (12, 6))
    plt.subplot(121)
    plt.plot(freqs[::1], [abs(a)/AmplitudeSum for a in X[:len(freqs):1]])
    #plt.plot([a[0] for a in currentAmplitudesByFreq[::1]], [a[1] for a in currentAmplitudesByFreq[::1]])
    plt.xlabel('Freq (Hz)')
    plt.ylabel('FFT Amplitude |X(freq)|')
    plt.subplot(122)
    plt.plot([a[0] for a in avgAmplitudesByFreq[::1]], [a[1] for a in avgAmplitudesByFreq[::1]])
    plt.xlabel('Freq (Hz)')
    plt.ylabel('FFT Amplitude |X(freq)|')
    plt.tight_layout()
    plt.show()
    #

    clock = time.time()

# sample mixing signal



# Exporting
with open('../results({})({}).csv'.format(numberOfSongsAdded, date.today), 'w') as f:
    output = csv.writer(f)
    output.writerow(avgAmplitudesByFreq)

# 

