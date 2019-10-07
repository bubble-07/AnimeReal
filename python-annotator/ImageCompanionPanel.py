import wx
import Parameters
from ImageViewPanel import *
from IdentifiedPoint import *
import wx.lib.scrolledpanel as scrolled

#Companion to a BodyDisplayPanel -- this displays
#a captured RGB image from the Kinect, for annotation
class ImageCompanionPanel(scrolled.ScrolledPanel):
    #TODO: Allow frame scrubbing here, and in the BodyDisplayPanel!
    def __init__(self, image, parent):
        scrolled.ScrolledPanel.__init__(self, parent, -1)

        self.image_panel = ImageViewPanel(image, self)

        #hbox = wx.BoxSizer(wx.VERTICAL)
        #hbox.Add(self.image_panel, wx.ID_ANY, wx.ALIGN_LEFT | wx.ALL, 1)
        #self.SetSizer(hbox)
        grid = wx.GridSizer(1,1,1,1)
        grid.Add(self.image_panel, wx.ID_ANY, wx.ALIGN_LEFT | wx.ALL, 1)
        self.SetSizer(grid)

        self.image_panel.Bind(wx.EVT_LEFT_UP, self.leftClickListener)
        self.image_panel.Bind(wx.EVT_RIGHT_UP, self.rightClickListener)
        self.image_panel.Bind(wx.EVT_KEY_DOWN, self.keyListener)
        self.newPoint = None
        self.SetupScrolling()
    def setAnnotationManager(self, annotation_manager):
        self.annotationManager = annotation_manager
    def getAnnotationManager(self):
        return self.annotationManager

    def setImage(self, image):
        self.image_panel.setImage(image)

    def keyListener(self, evt):
        key = evt.GetKeyCode()
        #TODO: deduplicate this code
        if (key in [wx.WXK_LEFT, wx.WXK_RIGHT]):
            if key == wx.WXK_LEFT:
                self.getAnnotationManager().scrub(-Parameters.scrub_speed)
            if key == wx.WXK_RIGHT:
                self.getAnnotationManager().scrub(Parameters.scrub_speed)
        if (key is ord('P')):
            self.getAnnotationManager().saveAssociationsToFile()
            print "Saved annotations to file"
    

    def OnChildFocus(self, event):
        event.Skip()

    #Sets the panel which is the companion to this one
    def setCompanion(self, companion):
        self.companion = companion
    def getMarkedPoints(self):
        return self.image_panel.getMarkedPoints()
    def hasNewPoint(self):
        return self.newPoint is not None
    def getNewPoint(self):
        return self.newPoint
    def newPointPaired(self):
        self.newPoint = None
    def removePoint(self, point):
        self.image_panel.removePoint(point)
    def clearPoints(self):
        self.newPoint = None
        self.image_panel.clearPoints()
    def addPoint(self, point):
        self.image_panel.addPoint(point)

    def leftClickListener(self, evt):
        #TODO: Remove!
        #Print the color of the clicked point
        print "Color (RGB): ", self.image_panel.getColorAt(evt.GetPosition())


        #If we already have a new point for pairing, clear the old one
        #and set the point to something different
        if (self.hasNewPoint()):
            oldPoint = self.getNewPoint()
            self.removePoint(oldPoint)
            mousePoint = IdentifiedPoint(evt.GetPosition())
            self.newPoint = mousePoint
            self.image_panel.addPoint(mousePoint)
            return

        #Otherwise, we must be in the mood for pairing!
        if (self.companion.hasNewPoint()):
            #We must be trying to pair with the companion's existing point!
            pairedIdedPoint = self.companion.getNewPoint()
            mousePoint = pairedIdedPoint.createLinked(evt.GetPosition())
            self.image_panel.addPoint(mousePoint)
            self.companion.newPointPaired()
        else:
            #We must be the first thing specified in the pairing
            self.newPoint = IdentifiedPoint(evt.GetPosition())
            self.image_panel.addPoint(self.newPoint)

    def rightClickListener(self, evt):
        closestPoint = self.image_panel.getClosestPoint(evt.GetPosition())
        if (closestPoint is None):
            return
        #Determine if "closestPoint" is in fact the newest-added point
        if (self.hasNewPoint() and closestPoint.hasSameId(self.getNewPoint())):
            #If so, clear our newPoint field
            self.newPoint = None
        self.companion.removePoint(closestPoint)
        self.image_panel.removePoint(closestPoint)
        
