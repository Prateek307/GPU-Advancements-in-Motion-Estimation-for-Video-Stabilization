import cv2
import tqdm
import numpy as np
from numpy import nan
from scipy import signal
from numpy.linalg import pinv, cond
from helpers import *
from kernel_functions import convolve2d_opencl, pinv_opencl

def LucasKanadeOpticalFlow(I1, I2, dev="CPU"):
    I1 = np.array(I1)
    I2 = np.array(I2)
    S = np.shape(I1)

    # if(dev=="GPU"):        
    #     Ix = convolve2d_opencl(I1, np.array([[-0.25, 0.25], [-0.25, 0.25]])) + convolve2d_opencl(I2, np.array([[-0.25, 0.25], [-0.25, 0.25]]))
    #     Iy = convolve2d_opencl(I1, np.array([[-0.25, -0.25], [0.25, 0.25]])) + convolve2d_opencl(I2, np.array([[-0.25, -0.25], [0.25, 0.25]]))
    #     It = convolve2d_opencl(I1, np.array([[0.25, 0.25], [0.25, 0.25]])) + convolve2d_opencl(I2, np.array([[-0.25, -0.25], [-0.25, -0.25]]))
    # else:
        # Ix = signal.convolve2d(I1, [[-0.25, 0.25], [-0.25, 0.25]], 'same') + signal.convolve2d(I2, [[-0.25, 0.25], [-0.25, 0.25]], 'same')
        # Iy = signal.convolve2d(I1, [[-0.25, -0.25], [0.25, 0.25]], 'same') + signal.convolve2d(I2, [[-0.25, -0.25], [0.25, 0.25]], 'same')
        # It = signal.convolve2d(I1, [[0.25, 0.25], [0.25, 0.25]], 'same') + signal.convolve2d(I2, [[-0.25, -0.25], [-0.25, -0.25]], 'same')
        
    Ix = signal.convolve2d(I1, [[-0.25, 0.25], [-0.25, 0.25]], 'same') + signal.convolve2d(I2, [[-0.25, 0.25], [-0.25, 0.25]], 'same')
    Iy = signal.convolve2d(I1, [[-0.25, -0.25], [0.25, 0.25]], 'same') + signal.convolve2d(I2, [[-0.25, -0.25], [0.25, 0.25]], 'same')
    It = signal.convolve2d(I1, [[0.25, 0.25], [0.25, 0.25]], 'same') + signal.convolve2d(I2, [[-0.25, -0.25], [-0.25, -0.25]], 'same')
    
    features = cv2.goodFeaturesToTrack(I1, 10000, 0.01, 10)
    features = np.intp(features)
    u = v = np.ones((S))

    for l in features:
        j, i = l.ravel()
        IX = [Ix[i - 1, j - 1], Ix[i, j - 1], Ix[i - 1, j - 1], Ix[i - 1, j], Ix[i, j], Ix[i + 1, j], Ix[i - 1, j + 1], Ix[i, j + 1], Ix[i + 1, j - 1]]
        IY = [Iy[i - 1, j - 1], Iy[i, j - 1], Iy[i - 1, j - 1], Iy[i - 1, j], Iy[i, j], Iy[i + 1, j], Iy[i - 1, j + 1], Iy[i, j + 1], Iy[i + 1, j - 1]]
        IT = [It[i - 1, j - 1], It[i, j - 1], It[i - 1, j - 1], It[i - 1, j], It[i, j], It[i + 1, j], It[i - 1, j + 1], It[i, j + 1], It[i + 1, j - 1]]

        # using the minimum least square solution approach
        LK = (IX, IY)
        LK = np.matrix(LK)
        LK_T = np.array(np.matrix(LK))
        LK = np.array(np.matrix.transpose(LK))

        # Pseudo Inverse
        A1 = np.dot(LK_T, LK)
        if(dev=="CPU"):
            A2 = pinv(A1)
        else:
            A2 = pinv_opencl(A1)
        A3 = np.dot(A2, LK_T)
        u[i, j], v[i, j] = np.dot(A3, IT)

    u = np.flipud(u)
    v = np.flipud(v)
    return u, v

def IterativeProcess(I1, I2, u1, v1, dev):
    I1 = np.array(I1)
    I2 = np.array(I2)
    S = np.shape(I1)

    u1 = np.round(u1)
    v1 = np.round(v1)

    u = np.zeros(S)
    v = np.zeros(S)


    for i in tqdm.tqdm(range(2, S[0]-2), desc="Refining Flow"):
        for j in range(2, S[1]-2):
            I1new = I1[i - 2:i + 3, j - 2:j + 3]
            lr = (i - 2) + v1[i, j]
            hr = (i + 2) + v1[i, j]
            lc = (j - 2) + u1[i, j]
            hc = (j + 2) + u1[i, j]

            # window search and selecting the last window,
            #  if it goes out of bounds
            if lr < 0:
                lr = 0
                hr = 4
            if lc < 0:
                lc = 0
                hc = 4
            if hr > (len(I1[:, 0])) - 1:
                lr = len(I1[:, 0]) - 5
                hr = len(I1[:, 0]) - 1
            if hc > (len(I1[0, :])) - 1:
                lc = len(I1[0, :]) - 5
                hc = len(I1[0, :]) - 1
            if np.isnan(lr):
                lr = i - 2
                hr = i + 2
            if np.isnan(lc):
                lc = j - 2
                hc = j + 2

            I2new = I2[int(lr):int((hr + 1)), int(lc):int(hc + 1)]

            # if(dev=="GPU"):
            #     Ix = convolve2d_opencl(I1new, np.array([[-0.25, 0.25], [-0.25, 0.25]])) + convolve2d_opencl(I2new, np.array([[-0.25, 0.25], [-0.25, 0.25]]))
            #     Iy = convolve2d_opencl(I1new, np.array([[-0.25, -0.25], [0.25, 0.25]])) + convolve2d_opencl(I2new, np.array([[-0.25, -0.25], [0.25, 0.25]]))
            #     It = convolve2d_opencl(I1new, np.array([[0.25, 0.25], [0.25, 0.25]])) + convolve2d_opencl(I2new, np.array([[-0.25, -0.25], [-0.25, -0.25]]))
            # else:
            #     Ix = signal.convolve2d(I1new, [[-0.25, 0.25], [-0.25, 0.25]], 'same') + signal.convolve2d(I2new, [[-0.25, 0.25], [-0.25, 0.25]], 'same')
            #     Iy = signal.convolve2d(I1new, [[-0.25, -0.25], [0.25, 0.25]], 'same') + signal.convolve2d(I2new, [[-0.25, -0.25], [0.25, 0.25]], 'same')
            #     It = signal.convolve2d(I1new, [[0.25, 0.25], [0.25, 0.25]], 'same') + signal.convolve2d(I2new, [[-0.25, -0.25], [-0.25, -0.25]], 'same')
            Ix = signal.convolve2d(I1new, [[-0.25, 0.25], [-0.25, 0.25]], 'same') + signal.convolve2d(I2new, [[-0.25, 0.25], [-0.25, 0.25]], 'same')
            Iy = signal.convolve2d(I1new, [[-0.25, -0.25], [0.25, 0.25]], 'same') + signal.convolve2d(I2new, [[-0.25, -0.25], [0.25, 0.25]], 'same')
            It = signal.convolve2d(I1new, [[0.25, 0.25], [0.25, 0.25]], 'same') + signal.convolve2d(I2new, [[-0.25, -0.25], [-0.25, -0.25]], 'same')

            IX = np.transpose(Ix[1:5, 1:5])
            IY = np.transpose(Iy[1:5, 1:5])
            IT = np.transpose(It[1:5, 1:5])

            IX = IX.ravel()
            IY = IY.ravel()
            IT = IT.ravel()

            LK = (IX, IY)
            LK = np.matrix(LK)
            LK_T = np.array(np.matrix(LK))
            LK = np.array(np.matrix.transpose(LK))

            A1 = np.dot(LK_T, LK)            
            if(dev=="CPU"):
                A2 = pinv(A1)
            else:
                A2 = pinv_opencl(A1)
                          
            A3 = np.dot(A2, LK_T)
            (u[i, j], v[i, j]) = np.dot(A3, IT)

    r = np.mat(np.transpose(LK))*np.mat(LK)
    r = 1.0 / (cond(r))
    return u, v, r

def LK_Pyr(Im1, Im2, iteration, level, dev, visualize):
    I1 = np.array(Im1)
    I2 = np.array(Im2)
    S = np.shape(I1)

    pyramid1 = np.empty((S[0], S[1], level))
    pyramid2 = np.empty((S[0], S[1], level))
    pyramid1[:, :, 0] = I1
    pyramid2[:, :, 0] = I2

    for i in range(1, level):
        I1 = DownSample(I1)
        I2 = DownSample(I2)
        pyramid1[0:np.shape(I1)[0], 0:np.shape(I1)[1], i] = I1
        pyramid2[0:np.shape(I2)[0], 0:np.shape(I2)[1], i] = I2

    level0_I1 = pyramid1[0:int(len(pyramid1[:, 0]) / 4), 0:int(len(pyramid1[0, :]) / 4), 2]
    level0_I2 = pyramid2[0:int(len(pyramid2[:, 0]) / 4), 0:int(len(pyramid2[0, :]) / 4), 2]
    (u,v) = LucasKanadeOpticalFlow(Im1, Im2, dev)

    for i in range(0, iteration):
         (u, v, r)= IterativeProcess(level0_I1, level0_I2, u, v, dev)

    u_l0 = u
    v_l0 = v
    I_l0 = level0_I1
    u_l0[np.where(u_l0 == 0)] = nan
    v_l0[np.where(v_l0 == 0)] = nan

    k = 1
    u1 = UpSample(u)
    v1 = UpSample(v)
    I1new = pyramid1[0:int(len(pyramid1[:, 0]) / (2 ** (level - k - 1))), 0:int(len(pyramid1[0, :]) / (2 ** (level - k - 1))), level - k - 1]
    I2new = pyramid2[0:int(len(pyramid2[:, 0]) / (2 ** (level - k - 1))), 0:int(len(pyramid2[0, :]) / (2 ** (level - k - 1))), level - k - 1]
    u, v, r = IterativeProcess(I1new, I2new, u1, v1, dev)

    u_l1 = u
    v_l1 = v
    I_l1 = I1new
    u_l1[np.where(u_l1 == 0)] = nan
    v_l1[np.where(v_l1 == 0)] = nan

    k = 2
    u1 = UpSample(u)
    v1 = UpSample(v)
    I1new = pyramid1[0:int(len(pyramid1[:, 0]) / (2 ** (level - k - 1))), 0:int(len(pyramid1[0, :]) / (2 ** (level - k - 1))), level - k - 1]
    I2new = pyramid2[0:int(len(pyramid2[:, 0]) / (2 ** (level - k - 1))), 0:int(len(pyramid2[0, :]) / (2 ** (level - k - 1))), level - k - 1]
    u, v, r = IterativeProcess(I1new, I2new, u1, v1, dev)

    u_l2 = u
    v_l2 = v
    I_l2 = I1new
    u_l2[np.where(u_l2 == 0)] = nan
    v_l2[np.where(v_l2 == 0)] = nan

    # Plot the flow for visualization
    if(visualize):
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        plotOpticalFlow(axes[0], I_l0, u_l0, v_l0, 0)
        plotOpticalFlow(axes[1], I_l1, u_l1, v_l1, 1)
        plotOpticalFlow(axes[2], I_l2, u_l2, v_l2, 2)
        plt.savefig("pyramid.png")
        plt.show()

    return v_l2, u_l2