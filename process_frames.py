import cv2
import numpy as np

def process_video(path):
    cap = cv2.VideoCapture(path)

    cap.set(cv2.CAP_PROP_POS_FRAMES,1000-1)
    res,frame1 = cap.read()
    frame1 = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)

    cap.set(cv2.CAP_PROP_POS_FRAMES,1100-1)
    res,frame2 = cap.read()
    frame2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

    f1 = cv2.resize(frame1,(150,150),interpolation = cv2.INTER_LINEAR)
    f2 = cv2.resize(frame2,(150,150),interpolation = cv2.INTER_LINEAR)

    I1 = np.array(f1)
    I2 = np.array(f2)
    
    return I1, I2