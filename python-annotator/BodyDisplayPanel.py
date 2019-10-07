from StandardBodyView import *
import Parameters
from IdentifiedPoint import *
from BodyViewPanel import *
import wx
import os.path
import pickle

bodyViews = {}

#Doing this because it takes a long-ass time to generate the views!
bodyViewsFile = "BodyViews.pickle"
if (os.path.isfile("BodyViews.pickle")):
    bodyViews = pickle.load(open(bodyViewsFile, "rb"))
else:
    bodyViews = {
        "right" : StandardBodyView("right", [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]),
        "front" : StandardBodyView("front", [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], zclip=float('-inf')),
        "left" : StandardBodyView("left", [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
        "top" : StandardBodyView("top", [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]),
        "back" : StandardBodyView("back", [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], zclip=float('-inf')),
        "bottom" : StandardBodyView("bottom", [1.0, 0.0, 0.0], [0.0, 0.0, 1.0])}
    pickle.dump(bodyViews, open(bodyViewsFile, "wb"))



#Panel showing various displays of the body -- a companion
#to the ImageCompanionPanel
class BodyDisplayPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent, -1)
        vbox = wx.BoxSizer(wx.VERTICAL)
        hbox1 = wx.BoxSizer(wx.HORIZONTAL)
        hbox2 = wx.BoxSizer(wx.HORIZONTAL)
        vbox_bot_top = wx.BoxSizer(wx.VERTICAL)

        spacing = 1

        hpanel1 = wx.Panel(self)
        hpanel2 = wx.Panel(self)
        vbox_bot_top_panel = wx.Panel(hpanel2)
        vbox.Add(hpanel1, wx.ID_ANY, wx.EXPAND | wx.ALL, spacing)
        vbox.Add(hpanel2, wx.ID_ANY, wx.EXPAND | wx.ALL, spacing)

        right_panel = BodyViewPanel(bodyViews["right"], hpanel1)
        hbox1.Add(right_panel, wx.ID_ANY, wx.EXPAND | wx.ALL, spacing)

        front_panel = BodyViewPanel(bodyViews["front"], hpanel1)
        hbox1.Add(front_panel, wx.ID_ANY, wx.EXPAND | wx.ALL, spacing)

        left_panel = BodyViewPanel(bodyViews["left"], hpanel1)
        hbox1.Add(left_panel, wx.ID_ANY, wx.EXPAND | wx.ALL, spacing)

        top_panel = BodyViewPanel(bodyViews["top"], vbox_bot_top_panel)
        vbox_bot_top.Add(top_panel, wx.ID_ANY, wx.EXPAND | wx.ALL, spacing)

        bottom_panel = BodyViewPanel(bodyViews["bottom"], vbox_bot_top_panel)
        vbox_bot_top.Add(bottom_panel, wx.ID_ANY, wx.EXPAND | wx.ALL, spacing)
        hbox2.Add(vbox_bot_top_panel, wx.ID_ANY, wx.EXPAND | wx.ALL, spacing)

        back_panel = BodyViewPanel(bodyViews["back"], hpanel2)
        hbox2.Add(back_panel, wx.ID_ANY, wx.EXPAND | wx.ALL, spacing)

        #Collection of world position "markers"
        self.worldPositions = IdentifiedPointDict()

        self.panels = [right_panel, front_panel, left_panel, top_panel, bottom_panel, back_panel]
        for panel in self.panels:
            #Bind mouse listeners
            panelLeftClickListener = self.getLeftClickListener(panel)
            panel.Bind(wx.EVT_LEFT_UP, panelLeftClickListener)

            panelRightClickListener = self.getRightClickListener(panel)
            panel.Bind(wx.EVT_RIGHT_UP, panelRightClickListener)

            panel.Bind(wx.EVT_KEY_DOWN, self.keyListener)
        
        self.SetSizer(vbox)
        hpanel1.SetSizer(hbox1)
        hpanel2.SetSizer(hbox2)
        vbox_bot_top_panel.SetSizer(vbox_bot_top)
        self.newPoint = None
    #Sets the panel which is the companion to this one
    def setCompanion(self, companion):
        self.companion = companion

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
    
    def setAnnotationManager(self, annotation_manager):
        self.annotationManager = annotation_manager

    def getAnnotationManager(self):
        return self.annotationManager

    def hasNewPoint(self):
        return self.newPoint is not None
    def getNewPoint(self):
        return self.newPoint
    def newPointPaired(self):
        self.newPoint = None
    def get3dPoints(self):
        return self.worldPositions
    def removePoint(self, point):
        #Remove the point from each panel and from
        #the worldPositions dictionary
        for panel in self.panels:
            panel.removePoint(point)
        self.worldPositions.pop(point)

    def getLeftClickListener(self, bodyViewPanel):
        return (lambda evt: self.leftClickListener(bodyViewPanel, evt))
    def leftClickListener(self, bodyViewPanel, evt):
        #Find the 3d location of the click in the world
        worldPos = bodyViewPanel.get_world_pos(evt.GetPosition())
        if (worldPos is None):
            return

        print "World Position: ", worldPos

        #Okay, great. Now, we check what the pairing status
        #is with the companion.
        if (self.hasNewPoint()):
            #Replace the old unpaired point with a new one
            oldPoint = self.getNewPoint()
            self.removePoint(oldPoint)
            self.newPoint = IdentifiedPoint(worldPos)
            self.add3dPoint(self.newPoint)
            return

        #Otherwise, we must be in the mood for pairing!
        if (self.companion.hasNewPoint()):
            #We must be trying to pair with the companion
            pairedIdedPoint = self.companion.getNewPoint()
            #Create a linked 3d point with the companion
            mousePoint = pairedIdedPoint.createLinked(worldPos)
            self.add3dPoint(mousePoint)
            self.companion.newPointPaired()
        else:
            #We must be the first thing specified in the pairing
            self.newPoint = IdentifiedPoint(worldPos)
            self.add3dPoint(self.newPoint)

    def clearPoints(self):
        self.worldPositions = IdentifiedPointDict()
        self.newPoint = None
        for panel in self.panels:
            panel.clearPoints()


    def add3dPoint(self, newPoint):
        #Add it to the current marker list, and update all body view panels accordingly
        self.worldPositions.add(newPoint)

        for panel in self.panels:
            panel.tryAddProjectedPoint(newPoint)
        return newPoint

    
    def getRightClickListener(self, bodyViewPanel):
        return (lambda evt: self.rightClickListener(bodyViewPanel, evt))
    def rightClickListener(self, bodyViewPanel, evt):
        #Upon a right-click, within the bodyViewPanel, find the
        #closest identified point to the click
        closestPoint = bodyViewPanel.getClosestPoint(evt.GetPosition())
        if (closestPoint is None):
            #Must not be close enough! Ignore everything
            return
        #Determine if "closestPoint" is in fact the newest-added point
        if (self.hasNewPoint() and closestPoint.hasSameId(self.getNewPoint())):
            #If so, clear our newPoint field
            self.newPoint = None

        #Otherwise, go through each view panel and remove the points
        #that are identified with it
        self.removePoint(closestPoint)


