import numpy as np

#See https://github.com/CMU-Perceptual-Computing-Lab/panoptic-toolbox/blob/master/python/example.ipynb
#for _why_ this class carries so much info
class Camera(object):
    def __init__(self, camParamArray):
        #Camera distortion parameters
        self.K, self.distCoef, self.R, self.t = camParamArray
    #Method which takes a Camera and yields a flat representation of the camera parameters as floats
    def to_flat_rep(self):
        flat_K = self.K.flatten().squeeze()
        flat_distCoef = self.distCoef.flatten()
        flat_R = self.R.flatten().squeeze()
        flat_t = self.t.flatten()
        return np.concatenate((flat_K, flat_distCoef, flat_R, flat_t))
    #static method which takes the flat representation of camera parameters and constructs a Camera
    #object from that
    @staticmethod
    def from_flat_rep(flat_rep):
        flat_K = flat_rep[:9]
        flat_rep = flat_rep[9:]
        K = flat_K.reshape((3,3))

        flat_distCoef = flat_rep[:5]
        flat_rep = flat_rep[5:]
        distCoef = flat_distCoef

        flat_R = flat_rep[:9]
        flat_rep = flat_rep[9:]
        R = flat_R.reshape((3,3))

        flat_t = flat_rep
        t = flat_t.reshape((3, 1))

        return Camera([K, distCoef, R, t])


