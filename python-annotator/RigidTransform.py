from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import numpy as np
#Class representing a 3d rigid transformation in some
#relaive hierarchy of 3d rigid transformations

class RigidTransform:
    def __init__(self, parent=None, translation=np.array([0, 0, 0]),
                       rotation=np.identity(3)):
        self.parent = parent
        self.translation = np.copy(translation)
        self.rotation = np.copy(rotation)
    def fromMatrix4f(parent, matrix):
        return RigidTransform(parent).setMatrix4f(matrix)
    def setMatrix4f(self, matrix):
        self.translation = np.copy(matrix[3, :3])
        self.rotation = np.copy(matrix[:3, :3])
        return self
    #Derives a rigid transform which is a translated version of this one, and
    #has this as a parent
    def childTranslate(self, translation):
        return RigidTransform(parent=self, translation=translation, rotation=np.identity(3))

    #RigidTransforms represent transforms as T*R for R a rotation matrix and T a translation ma
    #However, if we have a matrix expressed the other way around, as R*T, to convert it to righ
    #format, the upper 3x3 rotation matrix should stay the same as R, but the translation part 
    #the resulting matrix will be set to t*transpose(R) for the translation vector 
    @staticmethod
    def twiddleTranslation(rotation, t):
        return np.matmul(t, rotation)

    def getRootTransform(self):
        if (self.parent is None):
            return self
        else:
            return self.parent.getRootTransform()

    def getTransformFromRoot(self):
        if (self.parent is None):
            return self
        prevTransforms = self.parent.getTransformFromRoot()
        prevTranslation = prevTransforms.getTranslation()
        prevRotation = prevTransforms.getRotation()
        myTranslation = self.getTranslation()
        myRotation = self.getRotation()

        twiddledTranslation = RigidTransform.twiddleTranslation(prevRotation, myTranslation)
        resultTranslation = prevTranslation + twiddledTranslation
        resultRotation = RigidTransform.normalize(np.matmul(prevRotation, myRotation))
        return RigidTransform(self.getRootTransform(), resultTranslation, resultRotation)
    
    @staticmethod
    def normalize(matrix):
        #TODO: Implement me!
        return matrix

    #TODO: Move this crap somewhere else!
    @staticmethod
    def transpose3x3(matrix):
        #Transposes the upper 3x3 component of
        #the given matrix
        result = np.copy(matrix)
        result[:3, :3] = np.transpose(result[:3, :3])
        return result

    #Get the other transform from the perspective of this
    def getTransformTo(self, other):
        return other.getTransformFrom(self)
    
    #Get the transform from the perspective of other
    def getTransformFrom(self, other):
        otherFrom = other.getTransformFromRoot().getTheTransformFrom()
        thisTo = self.getTransformFromRoot().getTheTransformTo()
        #TODO: Normalize!
        return RigidTransform.normalize(np.matmul(otherFrom, thisTo))

    def enterGLTransformTo(self, other):
        self.enterGLTransform(self.getTransformTo(other))
    def enterGLTransformFrom(self, other):
        self.enterGLTransform(self.getTransformFrom(other))

    def getTranslation(self):
        return self.translation
    def getRotation(self):
        return self.rotation

    #Get a 4x4 matrix of just translation
    def getTranslationTo(self):
        x, y, z = self.translation
        return np.array([[1.0, 0.0, 0.0, 0.0],
                         [0.0, 1.0, 0.0, 0.0],
                         [0.0, 0.0, 1.0, 0.0],
                         [x, y, z, 1.0]])

    #Get a 4x4 matrix of just rotation
    def getRotationTo(self):
        result = np.zeros((4, 4))
        result[:-1, :-1] = self.rotation
        result[3, 3] = 1.0
        return result

    def getRotationFrom(self):
        return RigidTransform.transpose3x3(self.getRotationTo())
    def getTranslationFrom(self):
        result = np.copy(self.getTranslationTo())
        result[3, :3] = -result[3, :3]
        return result
    def getTheTransformTo(self):
        return np.matmul(self.getTranslationTo(), self.getRotationTo())

    def getTheTransformFrom(self):
        return np.matmul(self.getRotationFrom(), self.getTranslationFrom())

    @staticmethod
    def enterGLTransform(transform):
        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glMultMatrixf(transform)

    def enterTheGLTransformTo(self):
        RigidTransform.enterGLTransform(self.getTheTransformTo())
    def enterTheGLTransformFrom(self):
        RigidTransform.enterGLTransform(self.getTheTransformFrom())
    def exitGLTransformTo(self):
        glMatrixMode(GL_MODELVIEW)
        glPopMatrix()
    def exitGLTransformFrom(self):
        self.exitGLTransformTo()
