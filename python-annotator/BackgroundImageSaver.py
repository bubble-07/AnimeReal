#Script which takes a source folder containing unordered
#.jpgs and outputs in the destination folder a collection
#of .tfrecords in the background image format expected by the
#RGBTrainer

import sys
import cv2
import numpy as np
from BackgroundTFWriter import *
from glob import glob
import random


sourceRoot = sys.argv[1]
jpgFiles = [y for x in os.walk(sourceRoot) for y in glob(os.path.join(x[0], '*.jpg'))]
random.shuffle(jpgFiles)

destRoot = sys.argv[2]
writer = BackgroundTFWriter(destRoot)

#Okay, great. Now, for each jpeg in the source
while (len(jpgFiles) > 0):
    drawnJpeg = jpgFiles.pop()
    img = cv2.imread(drawnJpeg, cv2.IMREAD_COLOR)
    #Resize the image to 1430 by 952
    resized_img = cv2.resize(img, (1430, 952))
    writer.add(resized_img)
writer.flush()

