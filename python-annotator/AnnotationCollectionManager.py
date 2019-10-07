import msgpack
import os.path
from IdentifiedPoint import *

#Manages a collection of annotated common points
#between the template body (in 3d) and the rgb images of frames
#(in 2d). Provides the ability to save the annotations out to a
#msgpack formatted file

class AnnotationCollectionManager():
    def __init__(self, bodypanel, imgpanel, displaypanel, assocFilePath):
        self.bodypanel = bodypanel
        self.imgpanel = imgpanel
        self.displaypanel = displaypanel
        self.cloudManager = displaypanel.getCloudManager()
        self.assocFilePath = assocFilePath
        self.frameAssociationDict = {}
        if (os.path.isfile(assocFilePath)):
            #The file exists! Load annotations from there
            frameIndex = self.cloudManager.getFrameIndices()
            self.loadAssociationsFromFile()
            if (frameIndex in self.frameAssociationDict):
                associations = self.frameAssociationDict[frameIndex]
                self.loadAssociations(associations)
    def loadAssociationsFromFile(self):
        with open(self.assocFilePath, "rb") as f:
            message = f.read()
            unpacked = msgpack.unpackb(message, raw=False, use_list=False)
            self.frameAssociationDict = unpacked
    def saveAssociationsToFile(self):
        self.storeCurrentAnnotations()
        with open(self.assocFilePath, "wb") as f:
            to_write = msgpack.packb(self.frameAssociationDict, use_bin_type=True)
            f.write(to_write)
            

    def storeCurrentAnnotations(self):
        worldAnnotations = self.bodypanel.get3dPoints()
        imgAnnotations = self.imgpanel.getMarkedPoints()
        imgPoints = imgAnnotations.getValues()
        associations = []
        for imgPoint in imgPoints:
            try:
                worldPoint = worldAnnotations.get(imgPoint)
                imgX, imgY = imgPoint.getPoint()
                worldX, worldY, worldZ = worldPoint.getPoint()

                associations.append(([int(imgX), int(imgY)], [float(worldX), float(worldY), float(worldZ)]))
            except KeyError:
                #ignore it, it must not have been paired yet
                continue
        frameIndex = self.cloudManager.getFrameIndices()
        self.frameAssociationDict[frameIndex] = associations

    #Given a list of associations, reset the states
    #of the body display panel and the image panel
    #to include those associations, and nothing else
    def loadAssociations(self, associations):
        self.bodypanel.clearPoints()
        self.imgpanel.clearPoints()
        for imgPoint, worldPoint in associations:
            idedImgPoint = IdentifiedPoint(imgPoint)
            idedWorldPoint = idedImgPoint.createLinked(worldPoint)
            self.bodypanel.add3dPoint(idedWorldPoint)
            self.imgpanel.addPoint(idedImgPoint)

    def refreshAll(self):
        self.scrub(0)
        
    def scrub(self, num_frames):
        self.storeCurrentAnnotations()
        self.cloudManager.scrub(num_frames)
        #From the cloud manager, get the current rgb image
        rgb_img = self.cloudManager.getOrigFrame().getRGBForDisplay()
        self.imgpanel.setImage(rgb_img)

        frameIndex = self.cloudManager.getFrameIndices()
        #See if the indexed frame exists in the association dict
        associations = []
        if (frameIndex in self.frameAssociationDict):
            associations = self.frameAssociationDict[frameIndex]
        self.loadAssociations(associations)
        self.displaypanel.reloadPointCloud()

