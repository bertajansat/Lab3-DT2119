# DT2119, Lab 1 Feature Extraction

# Function given by the exercise ----------------------------------
import numpy as np
import matplotlib.pyplot as plt
import scipy
from lab1_tools import trfbank, lifter, tidigit2labels
from collections import defaultdict
import sklearn

def mspec(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, samplingrate=20000):
    """Computes Mel Filterbank features.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        samplingrate: sampling rate of the original signal

    Returns:
        N x nfilters array with mel filterbank features (see trfbank for nfilters)
    """
    frames = enframe(samples, winlen, winshift)
    preemph = preemp(frames, preempcoeff)
    windowed = windowing(preemph)
    spec = powerSpectrum(windowed, nfft)
    return logMelSpectrum(spec, samplingrate)

def mfcc(samples, winlen = 400, winshift = 200, preempcoeff=0.97, nfft=512, nceps=13, samplingrate=20000, liftercoeff=22):
    """Computes Mel Frequency Cepstrum Coefficients.

    Args:
        samples: array of speech samples with shape (N,)
        winlen: lenght of the analysis window
        winshift: number of samples to shift the analysis window at every time step
        preempcoeff: pre-emphasis coefficient
        nfft: length of the Fast Fourier Transform (power of 2, >= winlen)
        nceps: number of cepstrum coefficients to compute
        samplingrate: sampling rate of the original signal
        liftercoeff: liftering coefficient used to equalise scale of MFCCs

    Returns:
        N x nceps array with lifetered MFCC coefficients
    """
    mspecs = mspec(samples, winlen, winshift, preempcoeff, nfft, samplingrate)
    ceps = cepstrum(mspecs, nceps)
    return lifter(ceps, liftercoeff)

# Functions to be implemented ----------------------------------

def enframe(samples, winlen, winshift):
    """
    Slices the input samples into overlapping windows.

    Args:
        winlen: window length in samples.
        winshift: shift of consecutive windows in samples
    Returns:
        numpy array [N x winlen], where N is the number of windows that fit
        in the input signal
    """
    import math
    n_samples = len(samples)
    N = 1+((n_samples - winlen) // winshift)
    enframed = np.zeros((N,winlen))
    offset = 0
    for i in range(N):
        enframed[i]=samples[offset:offset+winlen]
        offset = offset + winshift
    return enframed


def preemp(input, p=0.97):
    """
    Pre-emphasis filter.

    Args:
        input: array of speech frames [N x M] where N is the number of frames and
               M the samples per frame
        p: preemhasis factor (defaults to the value specified in the exercise)

    Output:
        output: array of pre-emphasised speech samples
    Note (you can use the function lfilter from scipy.signal)
    """
    b = [1, -p]  # y[n]=b_0*x[n]+b_1*x[n-a]
    a = [1]
    preemph = np.zeros_like(input)
    for i in range(len(input)):
        preemph[i] = scipy.signal.lfilter(b,a, input[i])
    return preemph


def windowing(input):
    """
    Applies hamming window to the input frames.

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
    Output:
        array of windoed speech samples [N x M]
    Note (you can use the function hamming from scipy.signal, include the sym=0 option
    if you want to get the same results as in the example)
    """
    M = len(input[0])
    N = len(input)
    window = scipy.signal.windows.hamming(M,sym=False)
    #plt.plot(window)
    windowed = np.zeros_like(input)
    for i in range(N):
        windowed[i] = input[i] * window
    return windowed


def powerSpectrum(input, nfft):
    """
    Calculates the power spectrum of the input signal, that is the square of the modulus of the FFT

    Args:
        input: array of speech samples [N x M] where N is the number of frames and
               M the samples per frame
        nfft: length of the FFT
    Output:
        array of power spectra [N x nfft]
    Note: you can use the function fft from scipy.fftpack
    """
    fft = scipy.fftpack.fft(input, n=nfft)
    power_spectra = np.absolute(fft)**2
    return power_spectra

#TODO: answer question, verify!

def logMelSpectrum(input, samplingrate):
    """
    Calculates the log output of a Mel filterbank when the input is the power spectrum

    Args:
        input: array of power spectrum coefficients [N x nfft] where N is the number of frames and
               nfft the length of each spectrum
        samplingrate: sampling rate of the original signal (used to calculate the filterbank shapes)
    Output:
        array of Mel filterbank log outputs [N x nmelfilters] where nmelfilters is the number
        of filters in the filterbank
    Note: use the trfbank function provided in lab1_tools.py to calculate the filterbank shapes and
          nmelfilters
    """
    nfft = len(input[0])
    N = len(input)
    filter_bank = trfbank(samplingrate, nfft)
    nmelfilters = len(filter_bank)

    log_outputs = np.zeros((N,nmelfilters))
    for n in range(N):  # For each frame
        for m in range(nmelfilters):  # For each filter
            # producte escalar → energia del filtre
            energy = np.sum(input[n]*filter_bank[m])
            log_outputs[n, m] = np.log(energy + 1e-12)  # 1e-12 added to avoid log(0)

    return log_outputs

def cepstrum(input, nceps):
    """
    Calulates Cepstral coefficients from mel spectrum applying Discrete Cosine Transform

    Args:
        input: array of log outputs of Mel scale filterbank [N x nmelfilters] where N is the
               number of frames and nmelfilters the length of the filterbank
        nceps: number of output cepstral coefficients
    Output:
        array of Cepstral coefficients [N x nceps]
    Note: you can use the function dct from scipy.fftpack.realtransforms
    """
    N = len(input)
    cepstral_coefficients=np.zeros((N,nceps))
    for i in range(N):
        all_coefficients = scipy.fftpack.realtransforms.dct(input[i])
        cepstral_coefficients[i] = all_coefficients[:nceps] # Take only first nceps coefficients
    return cepstral_coefficients

# TODO: answer questions
def euclidean_dist(x,y):
    return  np.linalg.norm(x-y)

def dtw(x, y, dist):
    """Dynamic Time Warping.

    Args:
        x, y: arrays of size NxD and MxD respectively, where D is the dimensionality
              and N, M are the respective lenghts of the sequences
        dist: distance function (can be used in the code as dist(x[i], y[j]))

    Outputs:
        d: global distance between the sequences (scalar) normalized to len(x)+len(y)
        LD: local distance between frames from x and y (NxM matrix)
        AD: accumulated distance between frames of x and y (NxM matrix)
        path: best path through AD

    Note that you only need to define the first output for this exercise.
    """
    N=len(x)
    M=len(y)
    LD = np.zeros((N,M))
    AD = np.zeros((N,M))
    # Local euclidean distances matrix
    for i in range(N):
        for j in range(M): 
            LD[i][j]=dist(x[i],y[j])

            if i >= 1 and j >= 1:
                AD[i][j]=LD[i][j]+min(AD[i-1][j-1],AD[i][j-1],AD[i-1][j])
            elif i==0 and j >= 1:
                AD[i][j] = LD[i][j] + AD[i][j-1]
            elif i>=1 and j==0:
                AD[i][j] = LD[i][j] + AD[i-1][j]
            else:
                AD[i][j]=LD[i][j]
    d = AD[N-1][M-1] / (N + M)
    # Best path not implemented --> not necessary
    return d, LD, AD

