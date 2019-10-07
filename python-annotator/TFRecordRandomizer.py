#Given a source directory and a destination directory,
#take all autolabel tfrecords in the source and shuffle their contents (across files)
#into a new set of tfrecords in the destination directory

import sys
import os
from glob import glob
import numpy as np
import tensorflow as tf
import cv2
import numpy
from DepthTrainingTFReader import depthFrameLabelPairsFromTFRecord
import random
from DepthTrainingTFWriter import *

sequenceRoot = sys.argv[1]
tfrecordFiles = [y for x in os.walk(sequenceRoot) for y in glob(os.path.join(x[0], '*.tfrecord'))]

random.shuffle(tfrecordFiles)

#shuffle that list of paths

destRoot = sys.argv[2]

writer = DepthTrainingTFWriter(destRoot)

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
        fileContents = depthFrameLabelPairsFromTFRecord(tfrecordFile)
        random.shuffle(fileContents)
        drawnFileContents.append(fileContents)
    #Okay, great. Now, we repeatedly pop from a random list in drawnFileContents to determine
    #the next thing to write
    while (len(drawnFileContents) > 0):
        listInd = random.randrange(len(drawnFileContents)) 
        depthImage, templateIndexImage = drawnFileContents[listInd].pop()
        writer.add(depthImage, templateIndexImage)
        if (len(drawnFileContents[listInd]) == 0):
            drawnFileContents.pop(listInd)
    if (writer.getNumElemsBuffered() > 0):
        writer.flush()

