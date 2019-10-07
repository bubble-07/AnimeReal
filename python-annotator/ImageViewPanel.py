import wx
import cv2
import numpy as np
from IdentifiedPoint import *

#Panel for a given image display, with markers
#The BodyViewPanel inherits from this
class ImageViewPanel(wx.Panel):
    def __init__(self, image, parent):
        wx.Panel.__init__(self, parent, -1)
        self.parent = parent
        self.wx_bmp_panel = None
        self.setImage(image)
        self.marked_points = IdentifiedPointDict()
        self.remove_distance = 20.0

    def getColorAt(self, pos):
        x, y = pos
        return self.orig_img[y, x]

    def setImage(self, image):
        self.orig_img = image
        self.img = np.copy(self.orig_img)
        self.height, self.width, _ = self.img.shape
        self.wx_img = wx.EmptyImage(self.width, self.height)
        self.wx_img.SetData(self.img.tostring())
        self.wx_bmp = self.wx_img.ConvertToBitmap()
        if (self.wx_bmp_panel is None):
            self.wx_bmp_panel = wx.StaticBitmap(self, -1, self.wx_bmp, (0, 0))
        else:
            self.wx_bmp_panel.SetBitmap(self.wx_bmp)

 
    #Given a 2d position, return the closest identified point in the view
    def getClosestPoint(self, pos):
        return self.marked_points.getClosest(pos, self.remove_distance)

    #Given an identified point, remove it (if present in the view),
    #and return it
    def removePoint(self, idedPoint):
        result = self.marked_points.pop(idedPoint)
        if (result is None):
            return None
        self.refreshMarkers()
        return result

    def clearPoints(self):
        self.marked_points = IdentifiedPointDict()
        self.refreshMarkers()

    #Given an identified point, add it to the view and refresh markers
    def addPoint(self, idedPoint):
        self.marked_points.add(idedPoint)
        self.refreshMarkers()

    def getMarkedPoints(self):
        return self.marked_points

    #Refreshes the internal bitmap panel to draw markers
    def refreshMarkers(self):
        #Now, draw little circles at each one of the marker positions
        self.img = np.copy(self.orig_img)
        radius = 3
        color = (255, 255, 255)
        for marker in self.marked_points.getValues():
            x, y = marker.getPoint()
            cv2.circle(self.img, (x, y), radius, color)
        self.refreshDisplay()

    def refreshDisplay(self):
        self.wx_img.SetData(self.img.tostring())
        self.wx_bmp = self.wx_img.ConvertToBitmap()
        self.wx_bmp_panel.SetBitmap(self.wx_bmp)
        self.Refresh()

    def Bind(self, evtKind, handler):
        self.wx_bmp_panel.Bind(evtKind, handler)

