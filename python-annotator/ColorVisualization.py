#Silly thing used to visualize the distribution
#of colors in a collection of frames
from pprint import pprint
from functools import partial
import matplotlib
import Parameters

matplotlib.use('WX')
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt

import numpy as np
from FrameManager import *
import math
import numpy.random

import sys
from Frame import *

import PackedFrameLoader
import cv2

import wx

def visualize(X, ax):
        plt.cla()
        ax.scatter(X[:,0] ,  X[:,1], X[:, 2], color='red', label='Colors')
        plt.draw()
        plt.pause(0.001)
     
def randomRows(A, num):
    return A[np.random.choice(A.shape[0], num, replace=False), :]

sequenceRoot = sys.argv[1]

frameManager = FrameManager(sequenceRoot)

colorCollection = []

keepRows = 50

visited = 0

while True:
    frame = frameManager.getFrame()
    rgb = frame.getRGB()

    #rgb = cv2.GaussianBlur(rgb, (11, 11), 5)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    reshaped = hsv.reshape(-1, hsv.shape[-1])


    relevant = reshaped[:, 0] < 115.0
    relevant_two = reshaped[:, 0] < 160.0
    relevant_three = reshaped[:, 0] * 20.2353 + reshaped[:, 1] > 3515.59

    #relevant = np.logical_or(relevant, relevant_two)
    relevant = np.logical_or(relevant, relevant_three)
    
    relevant_reshaped = np.copy(reshaped)
    relevant_reshaped[relevant, :] = 0

    relevant_orig = np.copy(relevant_reshaped)

    relevant_reshaped = np.reshape(relevant_reshaped, hsv.shape)


    '''
    #TEST -- threshold out a given set of values 
    threshed = np.zeros(hsv.shape, dtype=np.uint8)
     

    for i in range(hsv.shape[0]):
        for j in range(hsv.shape[1]):
            h, s, v = hsv[i][j]
            if (v > 255 - s):
                threshed[i][j] = hsv[i][j]
    '''

    threshed_rgb = cv2.cvtColor(relevant_reshaped, cv2.COLOR_HSV2BGR)
    cv2.imshow('Silly', threshed_rgb)
    cv2.waitKey(1)

    kept = randomRows(relevant_orig, keepRows)

    for elem in kept:
        if (elem[0] > 0.1):
            colorCollection.append(elem)

    advanced = frameManager.tryAdvance(1)
    print visited
    visited += 1
    if (not advanced):
        break
colorCollection = np.array(colorCollection, dtype=np.float32)

#Great, now plot everything in the color collection as a scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.xlabel("H")
plt.ylabel("S")
visualize(colorCollection, ax)
plt.show()


