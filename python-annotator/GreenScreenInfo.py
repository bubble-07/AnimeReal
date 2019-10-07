#Simple script which crawls a folder containing several sequence roots
#and loads all background frames in sequence names containing "empty"
#then reports scatterplot graphs of Hue vs. Value, Hue vs. Saturation,
#and Saturation vs. Value

import sys
import numpy as np
from FrameManager import *
from RGBTrainingTFWriter import *
import matplotlib
from glob import glob
import matplotlib.pyplot as plt

def randomRows(A, num_rows):
    return A[np.random.choice(A.shape[0], num_rows, replace=False)]

if __name__ == '__main__':
    sequenceRootRoot = sys.argv[1]
    emptySequenceDirs = [y for x in os.walk(sequenceRootRoot) for y in glob(os.path.join(x[0], '*empty*/'))]
    cameraLabels = ["0", "1", "2"]
    rgbFrames = []
    for sequenceDir in emptySequenceDirs:
        for cam in cameraLabels:
            fullPath = os.path.join(sequenceDir, cam)
            if (os.path.isdir(fullPath)):
                frameManager = FrameManager(fullPath)
                LIMIT = 20
                i = 0
                while True:
                    frame = frameManager.getFrame()
                    #Great -- take all of the 
                    i += 1
                    if (i == LIMIT):
                        rgbFrames.append(frame.getRGB())
                        break
                    advanced = frameManager.tryAdvance(1)
                    if (not advanced):
                        break
    
    rgbFrames = np.reshape(np.array(rgbFrames), (-1, 3)).astype(np.float32) / 255.0
    
    NUM_TO_PLOT = 5000

    rgbFrames = randomRows(rgbFrames, NUM_TO_PLOT)

    #Great, now convert the whole thing to HSV
    hsvArray = matplotlib.colors.rgb_to_hsv(rgbFrames)
    hsvArray *= np.array([[179.0, 255.0, 255.0]], dtype=np.float32)


    #Okay, great. Now take the hsv array and generate the requisite plots
    hues = hsvArray[:, 0]
    sats = hsvArray[:, 1]
    values = hsvArray[:, 2]
    
    plt.subplot(131)    
    plt.scatter(hues, sats)

    plt.subplot(132)
    plt.scatter(sats, values)

    plt.subplot(133)
    plt.scatter(hues, values)
    plt.show()

