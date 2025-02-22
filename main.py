from pylab import *
import argparse
import pyopencl as cl
import cProfile
import pstats

from process_frames import process_video
from flow import LK_Pyr
from kernels import *
from ocl_init import print_platforms, get_device_info
from kernel_functions import DEVICE

VIDEO_PATH = './assets/video.mp4'

parser = argparse.ArgumentParser()
parser.add_argument("--use", choices=["CPU", "GPU"], default="CPU", help="Specify whether to use CPU or GPU: (CPU/GPU)")
parser.add_argument("--visualize", choices=["1", "0"], default="0", help="Specify whether visualize flow: (1/0)")

args = parser.parse_args()

if args.use == "GPU":
   dev = "GPU"
   platforms = cl.get_platforms()
   print_platforms(platforms)
   get_device_info(DEVICE)
else:
   dev = "CPU"

vis = True if (args.visualize=="1") else False

def main():
    # Get 2 frames from video
    f1, f2 = process_video(path=VIDEO_PATH)
    print("-"*60)
    print(f"Processing frames of size : {f1.shape}, {f2.shape}")
    print("-"*60)

    # Calculate Pyramidal Optical flow between frames
    u, v = LK_Pyr(f1, f2, 3, 3, dev, visualize=vis)

    print("The optical flow vectors: ")
    print(f"U: {u}")
    print(f"V: {v}")

if __name__ == '__main__':
  with cProfile.Profile() as profile:
    main()

    results = pstats.Stats(profile)
    results.sort_stats(pstats.SortKey.TIME)
    results.dump_stats("profiling_stats.dat")