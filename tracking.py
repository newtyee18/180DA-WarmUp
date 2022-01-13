from __future__ import print_function
import cv2 as cv
import numpy as np
import argparse
import random as rng
from matplotlib import pyplot as plt

cap = cv.VideoCapture(0)

lower1 = np.array([0,0,230])
upper1 = np.array([0,0,255])
lower2 = np.array([220,220,220])
upper2 = np.array([255,255,255])
kernelOpen = np.ones((7,7))
kernelClose = np.ones((20,20))

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    hsv_mask = cv.inRange(hsv, lower1, upper1)
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    rgb_mask = cv.inRange(rgb, lower2, upper2)
    
    mask3 = cv.morphologyEx (hsv_mask, cv.MORPH_OPEN, kernelOpen)
    mask4 = cv.morphologyEx (mask3, cv.MORPH_CLOSE, kernelClose)

    mask5 = cv.morphologyEx (rgb_mask, cv.MORPH_OPEN, kernelOpen)
    mask6 = cv.morphologyEx (mask5, cv.MORPH_CLOSE, kernelClose)
    
    #thresholding the frame
    ret,thresh1 = cv.threshold(gray,127,255,cv.THRESH_BINARY)

    contours, h = cv.findContours(hsv_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for c in contours:
        # get the bounding rect
        x, y, w, h = cv.boundingRect(c)
        # draw a green rectangle to visualize the bounding rect
        cv.rectangle(hsv, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv.drawContours(hsv, contours, -1, (255, 255, 0), 1)
    # Display the resulting frame
    cv.imshow('frame', hsv)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()