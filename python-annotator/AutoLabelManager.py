#Object responsible for managing automatically-generated
#"labels" (estimated annotations of where the template body
#points fit into the frame) corresponding to a given sequence
#(to be stored in a subfolder of the sequence folder called "autoLabels")
import os
from AutoLabelReader import *
from AutoLabelWriter import *
class AutoLabelManager():
    def __init__(self, sequencePath):
        self.path = os.path.join(sequencePath, "autoLabels")
        if (not os.path.isdir(self.path)):
            #Create the directory
            os.makedirs(self.path)
        self.setLastGathered()
    def setLastGathered(self):
        subNumbers = []
        for sub in os.listdir(self.path):
            if ".dat" in sub:
                subNumString = sub[0:-4]
                subNum = int(subNumString)
                subNumbers.append(subNum)
        if (len(subNumbers) == 0):
            #Must be no files!
            self.lastWrittenFileNumber = -1
            self.lastWrittenTimestamp = float('-inf')
            return 
        #Order the subfiles by number
        subNumbers.sort()
        #Great, now record the last one
        self.lastWrittenFileNumber = subNumbers[-1]
        #Also figure out the last gathered timestamp 
        lastFilePath = str(self.lastWrittenFileNumber) + ".dat"
        lastFilePath = os.path.join(self.path, lastFilePath)
        lastFileReader = AutoLabelReader(lastFilePath)        
        self.lastWrittenTimestamp = lastFileReader.getMaxTimestamp()
    def getLastWrittenTimestamp(self):
        return self.lastWrittenTimestamp
    def hasStartedWritingSequence(self):
        return (self.lastWrittenFileNumber != -1)
    #Given a MultiCloudManager, bring it to the position of one frame past
    #the first frame with the maximum timestamp
    #Return True iff the cloud manager has a frame like that
    def updateCloudManager(self, cloudManager):
        while True:
            if (cloudManager.getTimestamp() > self.lastWrittenTimestamp):
                break
            advance_success = cloudManager.scrub(1)  
            if (not advance_success):
                return False
        return True


    #Given an array of timestamps (presumably greater than the last written stamp)
    #get the AutoLabelWriter for the most recent file
    #After executing this operation, it's assumed that the file will
    #be written during the execution of the program, and hence, it is
    #safe to update the last written stamp
    def getLabelWriter(self, timestamps):
        lastTimestamp = timestamps[-1]
        self.lastWrittenTimestamp = lastTimestamp
        self.lastWrittenFileNumber += 1

        lastFilePath = str(self.lastWrittenFileNumber) + ".dat"
        lastFilePath = os.path.join(self.path, lastFilePath)
        lastFileWriter = AutoLabelWriter(lastFilePath)

        return lastFileWriter
