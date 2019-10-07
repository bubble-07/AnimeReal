#Convenient script to play back a visual representation of all of the
#autolabel files in a given directory
from RootDirector import *
from AutoLabelReader import *
import sys
import VizUtils
import os
import numpy as np
import StandardBody
from FrameManager import *
import cv2

def visualizePoints(points):
    #Given a point array in the same order as the standard body
    VizUtils.quickDisplayOrtho("MainWindow", points, StandardBody.standardColors)

def visualizeFrame(rgb):
    #First, resize to 512 x 424
    resized = cv2.resize(rgb, (424, 512))
    cv2.imshow("Frame", resized)
    cv2.waitKey(10)

sequenceRoot = sys.argv[1]
rootDirector = RootDirector(sequenceRoot)

for recordingName in rootDirector.getPeopleRecordingNames():
    recordingPath = os.path.join(rootDirector.getDirectoryPath(), recordingName)
    #Look inside the autoLabels path there
    autoLabelsPath = os.path.join(recordingPath, "autoLabels")
    recordingZeroCamPath = os.path.join(recordingPath, "0")
    frameManager = FrameManager(recordingZeroCamPath)
    subs = os.listdir(autoLabelsPath)
    subs.sort()
    for sub in subs:
        full = os.path.join(autoLabelsPath, sub)
        labelReader = AutoLabelReader(full)
        N = labelReader.getNumLabels()
        for i in range(N):
            points = labelReader.getLabel(i)
            timestamp = labelReader.getTimestamp(i)
            while (frameManager.getViewTime() < timestamp):
                frameManager.tryAdvance(1)
            visualizeFrame(frameManager.getFrame().getRGB())
            visualizePoints(points)
