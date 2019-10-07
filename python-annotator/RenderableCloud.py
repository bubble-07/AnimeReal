from OpenGL.GL import *
from OpenGL.GLUT import *

class RenderableCloud:
    #An OpenGL display list generated from a point cloud
    #TODO Camera identifiers here?
    def __init__(self, pointcloud):

        #To generate from a point cloud, we start building a display list
        self.dlist_ind = glGenLists(1)
        if not self.dlist_ind:
            print "Unable to create display list!"
            print glGetError()
        glColor3f(1, 1, 1)
        glNewList(self.dlist_ind, GL_COMPILE)

        glPointSize(2.0)

        #TODO: Okay to set point size outside of here?
        #Also okay to do this with transformations?

        glBegin(GL_POINTS)

        for point, color in pointcloud.pointIterator():
            x, y, z = point
            r, g, b, a = color
            r, g, b = (float(r) / 255.0, float(g) / 255.0, float(b) / 255.0)
            glColor3f(r, g, b)
            glVertex3f(x, y, z)

        glEnd()
        glEndList()


    def draw(self, alpha):
        glColor4f(1, 1, 1, alpha)
        glCallList(self.dlist_ind)

    
    #You MUST call cleanup() before these objects are gc'ed,
    #and may not use them to render after you call this
    def cleanup(self):
        glDeleteLists(self.dlist_ind, 1)

