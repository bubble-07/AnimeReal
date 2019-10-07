#Copy of AutoLabelTFRecordPlayer, but done for RGB training records
#TODO: Reduce copypasta
import sys
import os
import numpy as np
import tensorflow as tf
import cv2
import numpy
from OpenImagesTFReader import rgbAnnotationPairsFromTFRecord
from glob import glob

sequenceRoot = sys.argv[1]
#Get a recursive listing of all .tfrecord files
tfrecordFiles = [y for x in os.walk(sequenceRoot) for y in glob(os.path.join(x[0], '*.tfrecord'))]

for tfrecordFile in tfrecordFiles:
    rgbLabelPairs = rgbAnnotationPairsFromTFRecord(tfrecordFile)
    for rgbImage, annotationImage in rgbLabelPairs:

        annotationImage = annotationImage > 0
        annotationImage = annotationImage.astype(np.float32)

        rgbImage = cv2.cvtColor(rgbImage, cv2.COLOR_BGR2RGB)

        cv2.imshow("RGBImage", rgbImage)
        cv2.imshow("Annotation Image", annotationImage)
        cv2.waitKey(100)

