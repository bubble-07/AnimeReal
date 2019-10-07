#Same dealio as the TFRecordRandomizer, but for the RGB data
#TODO: Merge with TFRecordRandomizer! Only doing this for now
#because the technical debt is necessary for quicc prototyping

import sys
import os
from glob import glob
import numpy as np
import tensorflow as tf
import cv2
import numpy
from RGBTrainingTFReader import rgbFrameLabelPairsFromTFRecord
import random
from RGBTrainingTFWriter import *

sequenceRoot = sys.argv[1]
tfrecordFiles = [y for x in os.walk(sequenceRoot) for y in glob(os.path.join(x[0], '*.tfrecord'))]

random.shuffle(tfrecordFiles)

#shuffle that list of paths

destRoot = sys.argv[2]

writer = RGBTrainingTFWriter(destRoot)

NUM_SIMULTANEOUS_FILES=100

#Okay, great. Now what we do is we repeatedly draw up to NUM_SIMULTANEOUS_FILES, and load their contents

while (len(tfrecordFiles) > 0):
    print "-------------------------------------------"
    print "-------------------------------------------"
    print "TFRecord FILES REMAINING: ", len(tfrecordFiles)
    print "------------------------------------------"
    print "------------------------------------------"
    drawnFileContents = []
    while (len(tfrecordFiles) > 0 and len(drawnFileContents) < NUM_SIMULTANEOUS_FILES):
        tfrecordFile = tfrecordFiles.pop()
        fileContents = rgbFrameLabelPairsFromTFRecord(tfrecordFile)
        random.shuffle(fileContents)
        drawnFileContents.append(fileContents)
    #Okay, great. Now, we repeatedly pop from a random list in drawnFileContents to determine
    #the next thing to write
    while (len(drawnFileContents) > 0):
        listInd = random.randrange(len(drawnFileContents)) 
        if (len(drawnFileContents[listInd]) == 0):
            drawnFileContents.pop(listInd)
            continue
        depthImage, templateIndexImage = drawnFileContents[listInd].pop()
        writer.add(depthImage, templateIndexImage)
        if (len(drawnFileContents[listInd]) == 0):
            drawnFileContents.pop(listInd)
    if (writer.getNumElemsBuffered() > 0):
        writer.flush()

