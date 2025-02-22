import pyopencl as cl
import numpy as np
from scipy.signal import convolve2d
from main import VIDEO_PATH
from kernels import *
from kernel_functions import pinv_opencl, convolve2d_opencl
from process_frames import process_video

import cProfile
import pstats

Ix_kernel = np.array([[-0.25, 0.25], [-0.25, 0.25]])
f1, f2 = process_video(path=VIDEO_PATH)
 
matrix = np.array([[17705.375,  1707.125], [1707.125, 14077.375]], dtype=np.float64)

with cProfile.Profile() as profile:
    # Test OpenCL Kernel functions

    # Test for PINV
    res = pinv_opencl(matrix)
    print("Pinv result using OpenCL: ")
    print(res)
    # res2 = np.linalg.pinv(matrix)
    # print("Pinv result using numpy: ")
    # print(res2)

    # Test for convolution
    # output = convolve2d_opencl(f1, Ix_kernel)
    # print("Convolution result using OPENCL: ")
    # print(output)
    # print("Convolution result using Scipy: ")
    # output2 = convolve2d(f1, Ix_kernel, 'same')
    # print(output2)

    results = pstats.Stats(profile)
    results.sort_stats(pstats.SortKey.TIME)
    results.dump_stats("testing.dat")