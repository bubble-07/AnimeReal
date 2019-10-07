from pprint import pprint
from functools import partial
import matplotlib
import BodyFilters
import ColorFilters
from RootDirector import *
import Parameters
import scipy as sp
matplotlib.use('WX')
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
from pycpd import affine_registration
from pycpd import deformable_registration
from pycpd import rigid_registration
from StandardBodyView import *
from AnnotationCollectionManager import *
from FrameManager import *
from CloudManager import *
import Calibrator
import math
from BodyViewPanel import *
import DeformableReg
import numpy as np
import numpy.random
import time
import Rasterizer
import StandardBody
import sys
import cv2
from RigidTransform import *
from ImageCompanionPanel import *
from RGBPointCloud import *
from RenderableCloud import *
from ElasticRegistration import *
from NeighborlyPointCloud import *
from RenderableAxes import *
from Frame import *
import PackedFrameLoader

from wx import glcanvas

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *
print "importing body display panel"

from BodyDisplayPanel import *
print "body display panel imported"


print "importing wx"
import wx
print "wx imported"

def randomRows(A, num_rows):
    return A[np.random.choice(A.shape[0], num_rows, replace=False), :]

class ButtonPanel(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent, -1)

        box = wx.BoxSizer(wx.VERTICAL)
        box.Add((20, 30))
        c = MainCanvas(self)
        c.SetMinSize((1024,768))
        self.mainCanvas = c
        box.Add(c, 0, wx.ALIGN_CENTER|wx.ALL, 15)

        self.SetAutoLayout(True)
        self.SetSizer(box)
    def getMainCanvas(self):
        return self.mainCanvas

class MainCanvas(glcanvas.GLCanvas):
    def visualize(self, iteration, error, X, Y, ax):
        NUM_POINTS = 1000
        X = randomRows(X, NUM_POINTS)
        Y = randomRows(Y, NUM_POINTS)
        plt.cla()
        ax.scatter(X[:,0] ,  X[:,1], X[:, 2], color='red', label='Target')
        ax.scatter(Y[:,0] ,  Y[:,1], Y[:, 2], color='blue', label='Source')
        #plt.text(0.87, 0.92, 'Iteration: {:d}\nError: {:06.4f}'.format(iteration, error), horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize='x-large')
        ax.legend(loc='upper left', fontsize='x-large')
        plt.draw()
        plt.pause(0.001)
            
    def getCloudManager(self):
        return self.cloudManager
    def __init__(self, parent):
        glcanvas.GLCanvas.__init__(self, parent, -1)
        self.init = False
        self.context = glcanvas.GLContext(self)

        #The root rigid transform for all objects in the scene
        self.rootTransform = RigidTransform()

        self.cameraTransform = RigidTransform(parent=self.rootTransform)

        #Booleans for panning keypress events
        self.panning_back = False
        self.panning_forward = False
        self.panning_left = False
        self.panning_right = False
        self.panning_up = False
        self.panning_down = False

        #Speed multiplier for panning
        self.pan_speed = 10.0

        #Current absolute panning position
        self.pan_pos = np.array([0.0, 0.0, 0.0])
        
        #Current absolute spherical rotation (x and y components)
        self.angle = np.array([0.0, 0.0])

        #Proportional amount to rotate x and y by for every pixel
        self.rot_mul = 0.1

        self.axes = None

        #Current spherical radius
        self.spherical_radius = 1000.0

        #Zooming alters the spherical radius
        #at a rate proportional to the zoom speed
        self.zoom_speed = 1000.0

        #True if we're currently doing a click+drag rotation
        self.rotate_mode = False

        #Initial location for initiated drag events
        self.init_drag_pos = np.array([0.0, 0.0])

        self.current_drag_pos = np.array([0.0, 0.0])

        #Vertical fov for the perspective camera
        self.fovy = 50.0

        self.view_width = 1024
        self.view_height = 768
        self.aspect_ratio = float(self.view_width) / float(self.view_height)

        #Near and far clipping planes
        self.zNear = 0.01
        self.zFar = 10000.0

        #The current timestamp being viewed

        #To hold a list of Frames to display, sorted in increasing order of absolute time
        self.frames = []

        print "loading a lot of frames"


        
        #TODO: Add file dialog for this crap, don't do it here
        #The current loaded sequence's root directory string
        self.sequenceRoot = sys.argv[1]
        self.rootDirector = RootDirector(self.sequenceRoot)

        self.sequenceName = sys.argv[2]

        self.cloudManager = self.rootDirector.getMultiCloudManager(self.sequenceName)

        coloredTemplateFile = "ColoredTemplate.pickle"

        coloredBody = pickle.load(open(coloredTemplateFile, "rb"))

        #Filter down to just yellow arm (for now)

        coloredBody.indices = np.zeros((np.asarray(coloredBody.points).shape[0], 2))

        coloredBody.applyBodyFilter(BodyFilters.maskLeftUpperLeg, negated=True)
        #coloredBody.applyColorFilter(ColorFilters.maskTorso)
        #coloredBody.applyLargestComponentFilter()

        '''
        wallToViz = Calibrator.classifiedWalls[0]
        wallPoints, wallColors = wallToViz
        coloredBody.points = pcl.PointCloud(wallPoints)
        coloredBody.colors = wallColors
        '''

        '''
        calibFrame = self.rootDirector.getAverageCalibFrame("2")
        calibCloud = RGBPointCloud.fromFrame(calibFrame)
        calibCloud.applyColorFilter(ColorFilters.maskCalibCube, negated=True)
        '''


        self.cloudManager.setCloudOverride(coloredBody)

        #coloredBody.applyStatisticalFilter(mean_k=20, stdev_thresh=2.0)
        #self.cloudManager.pointcloud = coloredBody

        self.cloudManager.unsafeScrubUnderlying(90)
        


        self.renderablecloud = None #Initialized later

        print "colors set!"

        
        #TODO: Multiple cameras, and identifiers for each in Frame data

        self.size = None
        self.Bind(wx.EVT_ERASE_BACKGROUND, self.OnEraseBackground)
	self.Bind(wx.EVT_SIZE, self.OnSize)
        self.Bind(wx.EVT_PAINT, self.OnPaint)
        self.Bind(wx.EVT_LEFT_DOWN, self.OnMouseDown)
        self.Bind(wx.EVT_LEFT_UP, self.OnMouseUp)
        self.Bind(wx.EVT_MOTION, self.OnMouseMotion)
        self.Bind(wx.EVT_MOUSEWHEEL, self.OnMouseWheel)
        self.Bind(wx.EVT_KEY_DOWN, self.OnKeyDown)
        self.Bind(wx.EVT_KEY_UP, self.OnKeyUp)

    def setAnnotationManager(self, manager):
        self.annotationManager = manager

    def getAnnotationManager(self):
        return self.annotationManager

    def OnEraseBackground(self, event):
        pass # Do nothing, to avoid flashing on MSW.

    def OnSize(self, event):
        wx.CallAfter(self.DoSetViewport)
        event.Skip()

    def DoSetViewport(self):
        size = self.size = self.GetClientSize()
        self.SetCurrent(self.context)
        glViewport(0, 0, size.width, size.height)
        
    def OnPaint(self, event):
        dc = wx.PaintDC(self)
        self.SetCurrent(self.context)
        if not self.init:
            self.InitGL()
            self.init = True
        self.OnDraw()
        event.Skip()

    def handleKey(self, keyCode, valueToSet):
        if keyCode is ord('W'):
            self.panning_up = valueToSet
        if keyCode is ord('S'):
            self.panning_down = valueToSet
        if keyCode is ord('A'):
            self.panning_left = valueToSet
        if keyCode is ord('Q'):
            self.panning_back = valueToSet
        if keyCode is ord('E'):
            self.panning_forward = valueToSet
        if keyCode is ord('D'):
            self.panning_right = valueToSet

    def OnKeyDown(self, evt):
        key = evt.GetKeyCode()
        if (key is ord('P')):
            self.getAnnotationManager().saveAssociationsToFile()
            print "Saved annotations to file"
        if (key in [wx.WXK_LEFT, wx.WXK_RIGHT]):
            if key == wx.WXK_LEFT:
                self.changeFrame(-Parameters.scrub_speed)
            if key == wx.WXK_RIGHT:
                self.changeFrame(Parameters.scrub_speed)
        else:
            self.handleKey(key, True)
    def changeFrame(self, amount):
        manager = self.getAnnotationManager()
        manager.scrub(amount)

    def OnKeyUp(self, evt):
        key = evt.GetKeyCode()
        self.handleKey(key, False)

    def OnMouseDown(self, evt):
        self.rotate_mode = True
        self.CaptureMouse()
        x, y = evt.GetPosition()
        self.init_drag_pos = np.array([x, y])
        self.current_drag_pos = np.array([x, y])

    def OnMouseWheel(self, evt):
        if (evt.GetWheelRotation() is not 0):
            self.spherical_radius -= self.zoom_speed * evt.GetWheelRotation()
            self.Refresh(False)

    #Gets the rotational offset vecor (x and y) from the
    #initial drag position given a final drag position
    def getRotationalOffset(self):
        diff = self.current_drag_pos - self.init_drag_pos
        return self.rot_mul * diff

    def completeDrag(self):
        self.rotate_mode = False
        self.angle += self.getRotationalOffset()
    

    def OnMouseUp(self, evt):
        x, y = evt.GetPosition()
        self.current_drag_pos = np.array([x, y])
        self.completeDrag()
        self.ReleaseMouse()

    def OnMouseMotion(self, evt):
        if evt.Dragging() and evt.LeftIsDown():
            x, y = evt.GetPosition()
            self.current_drag_pos = np.array([x, y])
            self.Refresh(False)
    def InitDisplayables(self):
        self.axes = RenderableAxes()
        self.reloadPointCloud()

    def reloadPointCloud(self):
        if (self.renderablecloud is not None):
            #Free the display list for the cloud
            self.renderablecloud.cleanup()
        self.renderablecloud = RenderableCloud(self.cloudManager.getCloud())

    def InitGL(self):
        glEnable(GL_DEPTH_TEST)
        #glEnable(GL_LIGHTING)
        #glEnable(GL_LIGHT0)
        glClearColor(0, 0, 0, 0)
        self.InitDisplayables()
    def OnDraw(self):
        # clear color and depth buffers
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        #Adjust panning according to inputs
        if self.panning_right:
            self.pan_pos[0] -= self.pan_speed
        if self.panning_left:
            self.pan_pos[0] += self.pan_speed
        if self.panning_up:
            self.pan_pos[1] += self.pan_speed
        if self.panning_down:
            self.pan_pos[1] -= self.pan_speed
        if self.panning_forward:
            self.pan_pos[2] -= self.pan_speed
        if self.panning_back:
            self.pan_pos[2] += self.pan_speed


        #Now, if we're actively rotating,
        #the changes to angle haven't been committed yet, so generate
        #a display angle
        disp_angle = np.copy(self.angle)
        if self.rotate_mode:
            rot_offset = self.getRotationalOffset()
            disp_angle += rot_offset
        
        #Set the appropriate perspective matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(self.fovy, self.aspect_ratio, self.zNear, self.zFar)

        #Derive the current spherical view ransform
        #TODO: Automate this process of building 4x4 matrices
        #out of gl transforms
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        glPushMatrix()
        glTranslatef(0.0,  0.0, -self.spherical_radius)
        glRotatef(disp_angle[0], 0.0, 1.0, 0.0)
        glRotatef(disp_angle[1]+180, 1.0, 0.0, 0.0)
        glTranslatef(self.pan_pos[0], self.pan_pos[1], self.pan_pos[2])

        a = glGetFloatv(GL_MODELVIEW_MATRIX)
        spherical = np.reshape(np.array(a, dtype=np.float32), (4, 4))
        glPopMatrix()
        glLoadIdentity()

        self.cameraTransform.setMatrix4f(spherical)



        #Global scale transformation, so we can use millimeters for units, but see meters
        glScalef(0.001, 0.001, 0.001)

        #Render everything a 1/1000th scale
        axesTransform = self.rootTransform.childTranslate(self.pan_pos)

        axesTransform.enterGLTransformFrom(self.cameraTransform)
        self.axes.draw()
        axesTransform.exitGLTransformFrom()

        self.rootTransform.enterGLTransformFrom(self.cameraTransform)
        #Okay, render a renderable cloud
        self.renderablecloud.draw(1.0)
        self.rootTransform.exitGLTransformFrom()

        self.SwapBuffers()
        wx.CallLater(10, self.Refresh)

#----------------------------------------------------------------------
class RunDemoApp(wx.App):
    def __init__(self):
        wx.App.__init__(self, redirect=False)

    def OnInit(self):
        cframe = wx.Frame(None, -1, "Randemo: ", pos=(20,20), 
                          style=wx.DEFAULT_FRAME_STYLE, name="silly")
        cframe.Show(True)

        cpanel = BodyDisplayPanel(cframe)
        cframe.SetSize((1200 / 2,1700 / 2))
        self.cframe = cframe

        dimg = np.zeros((512, 512, 3), dtype=np.uint8)
        dframe = wx.Frame(None, -1, "Randemo: ", pos=(20,20), 
                          style=wx.DEFAULT_FRAME_STYLE, name="silly")
        dframe.Show(True)

        dpanel = ImageCompanionPanel(dimg, dframe)
        dframe.SetSize((1200 / 2,1700 / 2))
        self.dframe = dframe

        dpanel.setCompanion(cpanel)
        cpanel.setCompanion(dpanel)

        body_disp_panel = cpanel
        img_disp_panel = dpanel



        frame = wx.Frame(None, -1, "RunDemo: ", pos=(20, 20),
                        style=wx.DEFAULT_FRAME_STYLE, name="run a sample")
        #frame.CreateStatusBar()

        menuBar = wx.MenuBar()
        menu = wx.Menu()
        item = menu.Append(wx.ID_EXIT, "E&xit\tCtrl-Q", "Exit demo")
        self.Bind(wx.EVT_MENU, self.OnExitApp, item)
        menuBar.Append(menu, "&File")
        
        frame.SetMenuBar(menuBar)
        frame.Show(True)
        frame.Bind(wx.EVT_CLOSE, self.OnCloseFrame)

        win = ButtonPanel(frame)

        cloud_disp_panel = win.getMainCanvas()

        sequenceRoot = sys.argv[2]
        annotationFile = sequenceRoot + "/" + "annotations.dat"

        annotation_manager = AnnotationCollectionManager(body_disp_panel, 
                                    img_disp_panel, cloud_disp_panel, annotationFile)

        body_disp_panel.setAnnotationManager(annotation_manager)
        img_disp_panel.setAnnotationManager(annotation_manager)
        cloud_disp_panel.setAnnotationManager(annotation_manager)

        annotation_manager.refreshAll()

        # set the frame to a good size for showing the two buttons
        frame.SetSize((800,600))
        win.SetFocus()
        self.window = win
        frect = frame.GetRect()

        self.SetTopWindow(frame)
        self.frame = frame
        return True
        
    def OnExitApp(self, evt):
        self.frame.Close(True)

    def OnCloseFrame(self, evt):
        if hasattr(self, "window") and hasattr(self.window, "ShutdownDemo"):
            self.window.ShutdownDemo()
        evt.Skip()

app = RunDemoApp()
app.MainLoop()
