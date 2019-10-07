import StandardBody
import math
import Rasterizer
import numpy as np
#Holds information about a given orthographic
#viewing perspective of the standard body

class StandardBodyView():
    def __init__(self, name, xvec, yvec, zclip=0.0, mmPerPix=4.0):
        self.name = name
        self.xvec = np.array(xvec)
        self.zclip = zclip
        self.yvec = np.array(yvec)
        self.mmPerPix = mmPerPix
        self.generate_images()
    def getName(self):
        return self.name
    def generate_images(self):
        self.pointIndex, self.zVals, self.rasterizeParams = Rasterizer.rasterizeOrtho(StandardBody.pointCloud, self.xvec, self.yvec, self.mmPerPix, pzThrow=self.zclip)
        self.smoothedPointIndex = Rasterizer.smoothRaster(self.pointIndex, self.zVals)
        self.image = Rasterizer.colorizeRaster(self.pointIndex)
        self.smoothedImage = Rasterizer.colorizeRaster(self.smoothedPointIndex)
    def get_image(self):
        return self.image
    def get_smoothed_image(self):
        return self.smoothedImage
    def get_smoothed_index(self):
        return self.smoothedPointIndex

    #Projects a given 3d point to image index space
    def projectPoint(self, point):
        return self.rasterizeParams.project(point)

    def image_pos_helper(self, image, point, tol):
        height = len(image)
        width = len(image[0])
        
        #Determine the search region using the pointProjector
        likelyPointX, likelyPointY, _ = self.projectPoint(point)

        if (likelyPointX < 0 or likelyPointX >= width or
                likelyPointY < 0 or likelyPointY >= height):
            return None

        margin = int(math.ceil(tol / self.mmPerPix))
        low_i = max(likelyPointY - margin, 0)
        high_i = min(likelyPointY + margin, height - 1)
        low_j = max(likelyPointX - margin, 0)
        high_j = min(likelyPointX + margin, width - 1)
        

        closestPoint = None
        closestDistance = float('+inf')
        for i in range(low_i, high_i):
            for j in range(low_j, high_j):
                testPoint = image[i][j]
                if (testPoint is None):
                    continue
                distance = np.linalg.norm(point - testPoint)
                if (distance < closestDistance):
                    closestDistance = distance
                    closestPoint = [j, i]
        if (closestDistance < tol):
            return closestPoint
        return None


    #Given a 3d world coordinate, find the closest 2d [x, y] point
    #in image index space, but if the corresponding position
    #in 3d world coordinates isn't within the given tolerance,
    #return None
    def get_image_pos(self, point, tol=20.0):
        return self.image_pos_helper(self.pointIndex, point, tol)
    
    def get_smoothed_image_pos(self, point, tol=20.0):
        return self.image_pos_helper(self.smoothedPointIndex, point, tol)
        

    #Given a 2d [x, y] point in image index space,
    #return the 3d [x, y, z] world coordinate it corresponds to
    #or None if not present
    def get_world_pos(self, point):
        x, y = point
        try:
            return self.pointIndex[y][x]
        except IndexError:
            return None
    #Same as the above, but with the smoothed index
    def get_smoothed_world_pos(self, point):
        x, y = point
        try:
            return self.smoothedPointIndex[y][x]
        except IndexError:
            return None


