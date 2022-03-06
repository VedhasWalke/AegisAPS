from audioop import avg
import numpy as np
from scipy.io import wavfile
from os import listdir
from sox import file_info
import math
import time
import matplotlib.pyplot as plt
import csv

avgAmplitudesByFreq = [[a/10,0] for a in range(0,200000, 1)]
numberOfSongsAdded = 0 # number of songs currently in the averages of avgAmplitudesByFreq

clock = time.time()
for wavFileName in listdir('../resources/'):
    
    wavFilePath = '../resources/'+wavFileName
    if(file_info.duration(wavFilePath) <= 100): # discarding songs less than 100 seconds long
        continue
    sr, data = wavfile.read(wavFilePath)

    if(file_info.channels(wavFilePath) == 2): # stereo
        data1 = [a[1] for a in data]
        data = [a[0] for a in data]

        X = np.fft.fft(data1)
        N = len(X)
        n = np.arange(N)
        T = N/sr
        freqs = n/T
        freqs = [x for x in freqs if x < 20000]

        current_freq = -1
        for i in range(0,len(freqs)):
            if(math.floor(freqs[i]*10)/10 == current_freq):
                continue
            else:
                current_freq = math.floor(freqs[i]*10)/10
            
            avgAmplitudesByFreq[int(current_freq*10)][1] = (avgAmplitudesByFreq[int(current_freq*10)][1]*(numberOfSongsAdded)+abs(X[i])/N)/(numberOfSongsAdded+1) # abs to get magnitude from complex number result of fft, divide by N to normalize
        
        numberOfSongsAdded += 1
        print("Phase Complete ("+str(time.time()-clock)+")")

        # Plotting
        plt.figure(figsize = (12, 6))
        plt.subplot(121)

        plt.plot(freqs[::1], [abs(a)/N for a in X[:len(freqs):1]])
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
    
    X = np.fft.fft(data)
    N = len(X)
    n = np.arange(N)
    T = N/sr
    freqs = n/T
    freqs = [x for x in freqs if x < 20000]

    current_freq = -1
    for i in range(0,len(freqs)):
        if(math.floor(freqs[i]*10)/10 == current_freq):
            continue
        else:
            current_freq = math.floor(freqs[i]*10)/10
        
        avgAmplitudesByFreq[int(current_freq*10)][1] = (avgAmplitudesByFreq[int(current_freq*10)][1]*(numberOfSongsAdded)+abs(X[i])/N)/(numberOfSongsAdded+1) # abs to get magnitude from complex number result of fft, divide by N to normalize
        #assert avgAmplitudesByFreq[int(current_freq*10)][1] > 1
    
    numberOfSongsAdded += 1
    print("Phase Complete ("+str(time.time()-clock)+")")
    # Plotting
    plt.figure(figsize = (12, 6))
    plt.subplot(121)

    plt.plot(freqs[::1], [abs(a)/N for a in X[:len(freqs):1]])
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

# Exporting
with open('../results({}).csv'.format(numberOfSongsAdded), 'w') as f:
    output = csv.writer(f)
    output.writerow(avgAmplitudesByFreq)


