#Class used for writing a collection of autoLabels (see AutoLabelManager)
#into an autoLabel .dat file, which are of the (msgpack) format of a dictionary with fields:
#timestamps: an array of timestamps (actual Long)
#labels: an parallel array of numpy float32 arrays of size Nx3 in .tobytes() form, where N is the number of points
#in the standard template body
import msgpack
import numpy as np

class AutoLabelWriter():
    def __init__(self, out_filename):
        self.out_filename = out_filename
        self.timestamps = []
        self.labels = []
    #Add a label to the file we'll eventually write
    #takes a timestamp (Long) and a numpy float32 Nx3 array
    def add(self, timestamp, bodyPositions):

        bodyPositionBytes = bodyPositions.astype(np.float32).tobytes()

        self.timestamps.append(timestamp)
        self.labels.append(bodyPositionBytes)
    def writeToFile(self):
        dictToPack = {}
        dictToPack["timestamps"] = self.timestamps
        dictToPack["labels"] = self.labels
        with open(self.out_filename, 'wb') as labelFile:
            label_message = msgpack.packb(dictToPack, use_bin_type=True)
            labelFile.write(label_message)

