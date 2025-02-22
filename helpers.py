import cv2
import math
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def Gauss(x, sigma):
    if sigma == 0:
        return 0
    else:
        g = (1 / math.sqrt(2 * math.pi * sigma * sigma)) * math.exp(-x * x) / (2 * sigma * sigma)
        return g

def Gauss_Mask(sigma):
    g = []
    for i in range(-2, 3):
        g1 = Gauss(i, sigma)
        g2 = Gauss(i - 0.5, sigma)
        g3 = Gauss(i + 0.5, sigma)
        gaussian = (g1 + g2 + g3) / 3
        g.append(gaussian)
    return g

def DownSample(I):
    Ix = Iy = []
    I = np.array(I)
    S = np.shape(I)
    for i in range(S[0]):
        Ix.extend([signal.convolve(I[i, :], G, 'same')])
    Ix = np.array(np.matrix(Ix))
    Iy = Ix[::2, ::2]
    return Iy

def UpSample(I):
    I = np.array(I)
    S = np.shape(I)

    Ix = np.zeros((S[0],2*S[1]))
    Ix[:,::2] = I
    S1 = np.shape(Ix)
    Iy = np.zeros((2*S1[0],S1[1]))
    Iy[::2,:] = Ix
    Ig = cv2.GaussianBlur(Iy,(5,5),1.5,1.5)
    return Ig

def plotOpticalFlow(ax, I, u, v, level):
    ax.imshow(I, cmap='gray')
    ax.set_title(f'Level {level}')
    ax.quiver(u, v, color='b')

SIGMA = 1.5
G = Gauss_Mask(SIGMA)