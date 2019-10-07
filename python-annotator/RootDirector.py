#Play on words for an object responsible for crawling
#root directories of recording sessions.
#these have the format of containing a recording called
#"calib" containing frames for calibration, 
#"empty" containing frames for background subtraction,
#a file called "alignment.dat" containing camera
#extrinsic parameters required to register point
#clouds from multiple different cameras,
#and a miscellaneous collection of other folders
#containing recordings of interest

import os
from Calibrator import *
import CloudManager
from MultiCloudManager import *
from AutoLabelFrameReadManager import *
from Frame import *

class RootDirector():
    def __init__(self, directoryPath, init_calibrator=True, init_calib_frames=True, init_background_frames=True):
        self.directoryPath = directoryPath

        if (init_background_frames):
            self.loadAverageBackgroundFrames()
        else:
            self.averageBackgroundFrames = {}

        if (init_calib_frames):
            self.loadAverageCalibFrames()
        else:
            self.averageCalibFrames = {}


        if (init_calibrator):
            self.calibrator = Calibrator(self)
        else:
            self.calibrator = None
    def averageFrameDictHelper(self, prefix):
        result = {}
        backgroundPath = prefix
        for cameraLabel in self.getCameraLabels():
            cameraPath = os.path.join(backgroundPath, cameraLabel)
            cameraManager = FrameManager(cameraPath)
            averagedCameraFrame = Frame.averageFrames(cameraManager.getAllFrames())
            result[cameraLabel] = averagedCameraFrame
        return result
    def loadAverageBackgroundFrames(self):
        self.averageBackgroundFrames = self.averageFrameDictHelper(self.getEmptyFolderPath())
    def loadAverageCalibFrames(self):
        self.averageCalibFrames = self.averageFrameDictHelper(self.getCalibFolderPath())
    def getAverageBackgroundFrame(self, camera):
        return self.averageBackgroundFrames[camera]
    def getAverageCalibFrame(self, camera):
        return self.averageCalibFrames[camera]
    def getCalibrator(self):
        return self.calibrator

    def getCloudManager(self, sequenceFolder, cameraLabel, cameraMatrix=None):
        print "Getting cloud manager for sequence ", sequenceFolder, cameraLabel
        if (cameraMatrix is None):
            if (self.getCalibrator() is None):
                print "No calibrator, but calibrator wanted!"
            calibrator = self.getCalibrator()
            cameraMatrix = calibrator.getAlignmentMatrixForCamera(cameraLabel)

        backgroundFrame = self.getAverageBackgroundFrame(cameraLabel)
        cameraFrameManager = self.getFrameManager(sequenceFolder, cameraLabel)
        cameraCloudManager = CloudManager.CloudManager(backgroundFrame, cameraFrameManager, cloudTransform=cameraMatrix)
        return cameraCloudManager

    def getFrameManager(self, sequenceFolder, cameraLabel):
        basePath = os.path.join(self.directoryPath, sequenceFolder)
        cameraPath = os.path.join(basePath, cameraLabel)
        cameraFrameManager = FrameManager(cameraPath)
        return cameraFrameManager


    def getMultiCloudManager(self, sequenceFolder):
        cloudManagers = []
        for cameraLabel in self.getCameraLabels():
            cameraCloudManager = self.getCloudManager(sequenceFolder, cameraLabel)
            cloudManagers.append(cameraCloudManager)
        return MultiCloudManager(cloudManagers)

    def getAutoLabelFrameReadManager(self, sequenceFolder):
        basePath = os.path.join(self.directoryPath, sequenceFolder)
        autoLabelFolder = os.path.join(basePath, "autoLabels")  

        mcm = self.getMultiCloudManager(sequenceFolder)
        return AutoLabelFrameReadManager(mcm, autoLabelFolder)
        

    def getCalibFolderPath(self):
        return os.path.join(self.directoryPath, "calib")
    def getEmptyFolderPath(self):
        return os.path.join(self.directoryPath, "empty")
    def getAlignmentFilePath(self):
        return os.path.join(self.directoryPath, "alignment.dat")
    def getRecordingNames(self):
        result = []
        for sub in os.listdir(self.directoryPath):
            full = os.path.join(self.directoryPath, sub)
            if (os.path.isdir(full)):
                result.append(sub)
        return result
    def getDirectoryPath(self):
        return self.directoryPath
    def getPeopleRecordingNames(self):
        paths = self.getRecordingNames()
        result = []
        for path in paths:
            if "empty" in path:
                continue
            if "calib" in path:
                continue
            result.append(path)
        return result
    def getRecordingPaths(self):
        result = []
        for sub in os.listdir(self.directoryPath):
            full = os.path.join(self.directoryPath, sub)
            if os.path.isdir(full):
                result.append(full)
        return result
    #Get all recording paths that correspond to recordings of PEOPLE
    def getPeopleRecordingPaths(self):
        paths = self.getRecordingPaths()
        result = []
        for path in paths:
            if "empty" in path:
                continue
            if "calib" in path:
                continue
            result.append(path)
        return result
        

    def getCameraLabels(self):
        result = []
        for sub in os.listdir(self.getEmptyFolderPath()):
            if os.path.isdir(os.path.join(self.getEmptyFolderPath(), sub)):
                result.append(sub)
        return result
