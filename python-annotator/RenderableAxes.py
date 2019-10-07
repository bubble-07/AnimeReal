from OpenGL.GL import *
from OpenGL.GLUT import *
#Renderable x/y/z axes

class RenderableAxes:
    def __init__(self):
        #One tick for every 10cm (rendering units are mm)
        self.tick_spacing = 100.0
        #Ten ticks on each side, result is one meter radius
        self.num_ticks = 10
        self.axis_width = 5.0
        self.tick_width = 2.0
        self.tick_length = 0.5 * self.tick_spacing

    def draw(self):
        min_extent = -self.tick_spacing * self.num_ticks
        total_ticks = self.num_ticks * 2

        glLineWidth(self.axis_width)
        glColor4f(0, 1, 0, 1)
        glBegin(GL_LINES)
        #Draw x-axis
        glVertex3f(min_extent, 0, 0)
        glVertex3f(-min_extent, 0, 0)
        #Draw y-axis
        glVertex3f(0, min_extent, 0)
        glVertex3f(0, -min_extent, 0)
        #Draw z-axis
        glVertex3f(0, 0, min_extent)
        glVertex3f(0, 0, -min_extent)

        glEnd()

        glLineWidth(self.tick_width)
        glBegin(GL_LINES)

        #Draw x-axis ticks
        s = min_extent
        tick_length = self.tick_length
        tick_spacing = self.tick_spacing
        for i in range(total_ticks):
            glVertex3f(s, -tick_length, 0)
            glVertex3f(s, tick_length, 0)
            glVertex3f(s, 0, -tick_length)
            glVertex3f(s, 0, tick_length)
            s += tick_spacing
        #Draw y-axis ticks
        s = min_extent
        for i in range(total_ticks):
            glVertex3f(0, s, -tick_length)
            glVertex3f(0, s, tick_length)
            glVertex3f(-tick_length, s, 0)
            glVertex3f(tick_length, s, 0)
            s += tick_spacing
        #Draw z-axis ticks
        s = min_extent
        for i in range(total_ticks):
            glVertex3f(-tick_length, 0, s)
            glVertex3f(tick_length, 0, s)
            glVertex3f(0, -tick_length, s)
            glVertex3f(0, tick_length, s)
            s += tick_spacing
        glEnd()
