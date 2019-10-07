import wx
import cv2
import numpy as np
from ImageViewPanel import *
from IdentifiedPoint import *

#Panel for a given orthographic perspective of the body
class BodyViewPanel(ImageViewPanel):
    def __init__(self, bodyView, parent):
        image = bodyView.get_smoothed_image()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ImageViewPanel.__init__(self, image, parent)

        self.bodyView = bodyView
        name = bodyView.getName()

    #Given an identified 3d point, try to add a projected
    #2d version of the point, and update the display
    def tryAddProjectedPoint(self, idedPoint):
        point = idedPoint.getPoint()
        newPoint = self.bodyView.get_smoothed_image_pos(point)
        if (newPoint is None):
            return
        #Link the id
        newIdedPoint = idedPoint.createLinked(newPoint)
        #Add and refresh
        self.addPoint(newIdedPoint)

    #Given a 2d [x, y] point in image index space,
    #returns the 3d [x, y, z] world coordinate it corresponds to
    #or None if not present
    def get_world_pos(self, point):
        return self.bodyView.get_smoothed_world_pos(point)

