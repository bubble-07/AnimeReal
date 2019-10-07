#Copy of AutoLabelTFRecordPlayer, but done for RGB training records
#TODO: Reduce copypasta
import sys
import os
import numpy as np
import tensorflow as tf
import cv2
import numpy
from RGBTrainingTFReader import rgbFrameLabelPairsFromTFRecord
from glob import glob
import StandardBody

sequenceRoot = sys.argv[1]
#Get a recursive listing of all .tfrecord files
tfrecordFiles = [y for x in os.walk(sequenceRoot) for y in glob(os.path.join(x[0], '*.tfrecord'))]

standardColors = StandardBody.standardColors
standardColors = np.vstack((standardColors, np.array([0, 0, 0, 0], dtype=np.float32)))

for tfrecordFile in tfrecordFiles:
    rgbLabelPairs = rgbFrameLabelPairsFromTFRecord(tfrecordFile)
    for rgbImage, templateIndexImage in rgbLabelPairs:
        templateIndices = templateIndexImage.flatten()
        templateColors = standardColors[templateIndices]
        templateColors = np.reshape(templateColors, (424, 512, 4))
        templateColors = templateColors[:, :, 0:3]
        templateColors = templateColors.astype(np.uint8)
        templateColors = cv2.cvtColor(templateColors, cv2.COLOR_BGR2RGB)

        rgbImage = cv2.cvtColor(rgbImage, cv2.COLOR_BGR2RGB)

        cv2.imshow("TemplateColors", templateColors)
        cv2.imshow("RGBImage", rgbImage)
        cv2.waitKey(100)

