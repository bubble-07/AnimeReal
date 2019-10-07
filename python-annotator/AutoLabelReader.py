#Class used for reading AutoLabel files
import msgpack
import numpy as np
import StandardBody

class AutoLabelReader():
    def __init__(self, in_filename):
        self.in_filename = in_filename
        self.read_data()
    def read_data(self):
        with open(self.in_filename, 'rb') as labelFile:
            dictToUnpack = msgpack.unpackb(labelFile.read(), raw=False)
            self.timestamps = dictToUnpack["timestamps"]
            self.labels = dictToUnpack["labels"]
            #TODO: Raise a suitable error of some kind if the lengths don't match!
    def getMinTimestamp(self):
        return self.timestamps[0]
    def getNumLabels(self):
        return len(self.timestamps)
    def getMaxTimestamp(self):
        return self.timestamps[-1]
    def getTimestamp(self, i):
        return self.timestamps[i]
    #Returns a given entry as an Nx3 float32 numpy array
    def getLabel(self, i):
        binLabel = self.labels[i]
        N = StandardBody.pointArray.shape[0]
        flat_array = np.frombuffer(binLabel, dtype=np.float32)
        return np.reshape(flat_array, (N, 3))

