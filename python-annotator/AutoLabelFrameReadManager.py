#An object which manages the state of a MultiCloudManager
#to associate appropriate autolabeled frames.
#This object only iterates through frames for which
#there is autolabel data
import os
import numpy as np
from AutoLabelReader import *

class AutoLabelFrameReadManager():
    def __init__(self, multiCloudManager, labelDirectory):
        self.multiCloudManager = multiCloudManager
        self.labelDirectory = labelDirectory
        self.currentFileNumber = 0
        self.reloadLabelFile()
        self.updateCloudManager()
    def getLabelFilePath(self):
        fileName = str(self.currentFileNumber) + ".dat"
        filePath = os.path.join(self.labelDirectory, fileName)
        return filePath
    def reloadLabelFile(self):
        self.currentFileIndex = 0
        self.labelReader = AutoLabelReader(self.getLabelFilePath())
    #Try to advance one label into the future, returning False if this is not possible,
    #and True otherwise
    def advance(self):
        if (self.currentFileIndex + 1 < self.labelReader.getNumLabels()):
            #CASE: We're don't have to switch label files
            self.currentFileIndex += 1
            self.updateCloudManager()
            return True
        else:
            #Check if the next path would be a file
            self.currentFileNumber += 1
            if (not os.path.isfile(self.getLabelFilePath())):
                #No more files! Return False, and stay put!
                self.currentFileNumber -= 1
                return False
            else:
                self.reloadLabelFile()
                self.updateCloudManager()
                return True

    #Returns the labeling entry as a (num standard body points)x3 float32 numpy array
    def getLabel(self):
        return self.labelReader.getLabel(self.currentFileIndex)

    def getTimestamp(self):
        return self.labelReader.getTimestamp(self.currentFileIndex)

    def updateCloudManager(self):
        while (self.multiCloudManager.getTimestamp() < self.getTimestamp()):
            self.multiCloudManager.advance()

    def getCloud(self):
        return self.multiCloudManager.getCloud()

    def getOrigFrame(self):
        return self.multiCloudManager.getOrigFrame()

