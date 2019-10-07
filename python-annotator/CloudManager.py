from Frame import *
import PointTracking
import numpy as np
from RGBPointCloud import *
#An object responsible for managing (background-subracted) point clouds (and frames)
#gathered from a single camera
class CloudManager():
    def __init__(self, backgroundFrame, frameManager, cloudTransform=np.eye(4, dtype=np.float32)):
        self.backgroundFrame = backgroundFrame
        self.frameManager = frameManager
        self.cloudTransform = cloudTransform
        print cloudTransform
        self.reloadCloud()
    def getFrameIndices(self):
        return self.frameManager.getFrameIndices()
    def seekTo(self, fileNumber, frameNumber):
        self.frameManager.seekTo(fileNumber, frameNumber)
        self.reloadCloud()
    def scrub(self, num_frames):
        result = True
        if (num_frames > 0):
            result = self.frameManager.tryAdvance(num_frames)
        else:
            result = self.frameManager.tryRewind(-num_frames)
        self.reloadCloud()
        return result
    def getOrigCloud(self):
        return RGBPointCloud.fromFrame(self.origFrame)
    def getTimestamp(self):
        return self.origFrame.getTimestamp()
    def getOrigFrame(self):
        return self.origFrame
    def getFrame(self):
        return self.frame
    def getCloud(self):
        return self.pointcloud
    def reloadCloud(self):
        self.origFrame = self.frameManager.getFrame()

        self.frame = Frame.filterBackgroundOnDepth(self.origFrame, self.backgroundFrame)

        self.pointcloud = RGBPointCloud.fromFrame(self.frame)
        self.pointcloud.transform4x4(self.cloudTransform)
        self.pointcloud.applyBackgroundFilter()
        self.pointcloud.applyLargestComponentFilter()
        #PointTracking.filterCloudToPoints(self.pointcloud)
        self.pointcloud.voidIfMajorityGreen()
