#File containing color filters for particular body parts
import numpy as np
import matplotlib
import cv2

import math
import colorsys


#Given a pixel (rgb vector), evaluate maskFunc on it
def pixelSpaceMask(maskFunc, rgb):
    r, g, b = np.array([float(rgb[0]), float(rgb[1]), float(rgb[2])], dtype=np.float32) / 255.0
    h, s, v = colorsys.rgb_to_hsv(r, g, b)

    #Convert to OpenCV ranges
    hsvArray = np.array([[h * 179.0, s * 255.0, v * 255.0]], dtype=np.float32)

    truthValue = maskFunc(hsvArray)
    return truthValue[0]

def pixelSpaceVectorMask(maskFunc, rgb):
    rgb = np.array(rgb, dtype=np.float32) / 255.0

    hsvArray = matplotlib.colors.rgb_to_hsv(rgb)
    hsvArray *= np.array([[179.0, 255.0, 255.0]], dtype=np.float32)

    truthValues = maskFunc(hsvArray)
    return truthValues
    

#Given an image, returns a mask of the entire image
#using the given mask function (same shape as the original image,
#but binary
def imageSpaceMask(maskFunc, rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_BGR2HSV)
    reshaped = hsv.reshape(-1, hsv.shape[-1])

    relevant = maskFunc(reshaped)

    h, w, _ = hsv.shape
    return np.reshape(relevant, (h, w))

#Mask where everything but the calibration cube (in theory) is true
def maskCalibCube(reshaped):
    relevant = reshaped[:, 0] < 55.0
    relevant_two = reshaped[:, 0] > 100.0
    relevant = reshaped[:, 0] < 0
    relevant_two = reshaped[:, 0] > 180.0
    relevant_three = reshaped[:, 1] > 128.0
    relevant_four = reshaped[:, 2] > 100.0

    relevant = np.logical_or(relevant, relevant_two)
    relevant = np.logical_or(relevant, relevant_three)
    relevant = np.logical_or(relevant, relevant_four)
    return relevant

#Mask where everything but the red arm (in theory) is true
def maskRedArm(reshaped):

    relevant = reshaped[:, 0] < 160.0
    relevant_two = reshaped[:, 0] * 20.2353 + reshaped[:, 1] < 3515.59

    relevant = np.logical_or(relevant, relevant_two)
    return relevant

def maskRemnantYellow(reshaped):
    relevant = reshaped[:, 0] < 29.0
    relevant_two = reshaped[:, 0] > 40.0
    relevant = np.logical_or(relevant, relevant_two)
    return relevant

def maskRemnantWhite(reshaped):
    relevant = reshaped[:, 0] < 80.0
    relevant_two = reshaped[:, 0] > 105.0
    relevant_three = reshaped[:, 1] > 85.0
    relevant_four = reshaped[:, 2] < 224.0
    relevant = np.logical_or(relevant, relevant_two)
    relevant = np.logical_or(relevant, relevant_three)
    relevant = np.logical_or(relevant, relevant_four)
    return relevant

def maskRemnantRemnantWhite(reshaped):
    relevant = reshaped[:, 0] < 95.0
    relevant_two = reshaped[:, 0] > 99.0
    relevant_three = reshaped[:, 1] < 86.0
    relevant_four = reshaped[:, 1] > 113.0
    relevant_five = reshaped[:, 2] < 226.0
    relevant_six = reshaped[:, 2] > 245.0
    relevant = np.logical_or(relevant, relevant_two)
    relevant = np.logical_or(relevant, relevant_three)
    relevant = np.logical_or(relevant, relevant_four)
    relevant = np.logical_or(relevant, relevant_five)
    relevant = np.logical_or(relevant, relevant_six)
    return relevant

def maskRemnantRemnantRemnantWhite(reshaped):
    relevant = reshaped[:, 0] < 92
    relevant_two = reshaped[:, 0] > 105.0
    relevant_three = reshaped[:, 1] < 44.0
    relevant_four = reshaped[:, 1] > 148.0
    relevant_five = reshaped[:, 2] < 224.0
    relevant_six = reshaped[:, 2] > 250.0
    relevant = np.logical_or(relevant, relevant_two)
    relevant = np.logical_or(relevant, relevant_three)
    relevant = np.logical_or(relevant, relevant_four)
    relevant = np.logical_or(relevant, relevant_five)
    relevant = np.logical_or(relevant, relevant_six)
    return relevant

def maskRemnantRemnantYellow(reshaped):
    relevant = reshaped[:, 0] < 40
    relevant_two = reshaped[:, 0] > 46.0
    relevant_three = reshaped[:, 1] < 89.0
    relevant_four = reshaped[:, 1] > 148.0
    relevant_five = reshaped[:, 2] < 226.0
    relevant_six = reshaped[:, 2] > 255.0
    relevant = np.logical_or(relevant, relevant_two)
    relevant = np.logical_or(relevant, relevant_three)
    relevant = np.logical_or(relevant, relevant_four)
    relevant = np.logical_or(relevant, relevant_five)
    relevant = np.logical_or(relevant, relevant_six)
    return relevant


def maskRemnantRemnantRemnantYellow(reshaped):
    relevant = reshaped[:, 0] < 46
    relevant_two = reshaped[:, 0] > 54.0
    relevant_three = reshaped[:, 1] < 55.0
    relevant_four = reshaped[:, 1] > 90.0
    relevant_five = reshaped[:, 2] < 226.0
    relevant_six = reshaped[:, 2] > 255.0
    relevant = np.logical_or(relevant, relevant_two)
    relevant = np.logical_or(relevant, relevant_three)
    relevant = np.logical_or(relevant, relevant_four)
    relevant = np.logical_or(relevant, relevant_five)
    relevant = np.logical_or(relevant, relevant_six)
    return relevant




def maskTooSaturated(reshaped):
    relevant = reshaped[:, 1] > 150.0
    relevant_two = reshaped[:, 2] < 225
    relevant = np.logical_or(relevant, relevant_two)
    return relevant

def maskRedHand(reshaped):

    relevant = reshaped[:, 0] < 115.0
    relevant_two = reshaped[:, 0] * 20.2353 + reshaped[:, 1] > 3515.59

    relevant = np.logical_or(relevant, relevant_two)
    return relevant

def maskWhiteLegInTemplate(reshaped):
    relevant = reshaped[:, 2] < 200.0
    relevant_two = reshaped[:, 1] > 100.0
    relevant_three = reshaped[:, 0] < 43.0

    relevant = np.logical_or(relevant, relevant_two)
    relevant = np.logical_or(relevant, relevant_three)
    return relevant

#Due to sampling/averaging differences, it's sadly different if we're doing this
#on the point cloud or on the template
def maskWhiteLegInRGB(reshaped):

    relevant = reshaped[:, 2] < 180.0
    relevant_two = reshaped[:, 0] < 87.0
    relevant_three = reshaped[:, 0] > 105.0
    relevant_five = reshaped[:, 0] * (25.0 / 4.0) + reshaped[:, 1] > (3175.0 / 4.0)
    relevant_six = reshaped[:, 0] * 9 - reshaped[:, 1] > 645.0
    relevant_seven = reshaped[:, 0] * 11.0909 + reshaped[:, 2] < 1155.0
    relevant_eight = reshaped[:, 0] * 2.4186 - reshaped[:, 1] < 73.907
    relevant_nine = reshaped[:, 0] * 12 + reshaped[:, 1] > 1200.0
    relevant_ten = reshaped[:, 0] * 5.5 - reshaped[:, 1] > 524.5


    relevant = np.logical_or(relevant, relevant_two)
    relevant = np.logical_or(relevant, relevant_three)
    relevant = np.logical_or(relevant, relevant_five)
    relevant = np.logical_or(relevant, relevant_six)
    relevant = np.logical_or(relevant, relevant_seven)
    relevant = np.logical_or(relevant, relevant_eight)
    relevant = np.logical_or(relevant, relevant_nine)
    relevant = np.logical_or(relevant, relevant_ten)
    
    return relevant


def maskTorso(reshaped):

    relevant = reshaped[:, 0] > 115.0
    relevant_two = reshaped[:, 0] < 98.0
    relevant_three = reshaped[:, 1] < 150.0
    relevant_four = reshaped[:, 2] < 10.0


    relevant = np.logical_or(relevant, relevant_two)
    relevant = np.logical_or(relevant, relevant_three)
    relevant = np.logical_or(relevant, relevant_four)

    return relevant

def maskYellowHand(reshaped):


    relevant = reshaped[:, 0] > 43.0
    relevant_two = reshaped[:, 0] < 36.0
    relevant_three = reshaped[:, 1] < 30.0

    relevant = np.logical_or(relevant, relevant_two)
    relevant = np.logical_or(relevant, relevant_three)
    
    return relevant

def maskYellowArm(reshaped):

    relevant = reshaped[:, 0] > 35.0
    relevant_two = reshaped[:, 0] < 30.0
    relevant_three = reshaped[:, 1] < 60.0

    relevant = np.logical_or(relevant, relevant_two)
    relevant = np.logical_or(relevant, relevant_three)

    return relevant

def maskGreen(reshaped):
    relevant = reshaped[:, 0] < 56.0
    relevant_two = reshaped[:, 0] > 81.0
    return np.logical_or(relevant, relevant_two)

def nonTemplateMaskGreenScreen(reshaped):
    '''
    relevant = reshaped[:, 0] < 50.0
    relevant_two = reshaped[:, 0] > 90.0
    relevant_three = reshaped[:, 1] < 50.0
    relevant_four = reshaped[:, 2] < 60.0
    relevant_five = reshaped[:, 2] > 200.0
    '''

    relevant = reshaped[:, 0] < 20.0
    relevant_two = reshaped[:, 0] > 93.0
    relevant_three = reshaped[:, 1] < 48.0
    relevant_four = reshaped[:, 1] > 243.0
    #relevant_five = reshaped[:, 2] < 115.0
    relevant_five = reshaped[:, 2] < 70.0
    #relevant_six = reshaped[:, 2] > 252.0
    #relevant_six = reshaped[:, 2] > 238.0

    relevant = np.logical_or(relevant, relevant_two)
    relevant = np.logical_or(relevant, relevant_three)
    relevant = np.logical_or(relevant, relevant_four)
    relevant = np.logical_or(relevant, relevant_five)
    #relevant = np.logical_or(relevant, relevant_six)
    return relevant


#Everything that's green, except for the green leg
def maskGreenScreen(reshaped):
    
    relevant = maskGreen(reshaped)
    not_irrelevant = maskGreenLeg(reshaped)

    relevant = np.logical_not(relevant)
    relevant = np.logical_and(relevant, not_irrelevant)
    relevant = np.logical_not(relevant)
    return relevant



def maskGreenLeg(reshaped):

    relevant = reshaped[:, 1] * (23.0 / 18.0) + reshaped[:, 2] > 365
    relevant_two = reshaped[:, 2] > 170.0
    relevant_four = reshaped[:, 0] > 100
    relevant_five = reshaped[:, 1] < 100.0
    relevant_six = reshaped[:, 1] > 225.0
    relevant_seven = reshaped[:, 1] + 3.0 * reshaped[:, 0] < 350.0
    relevant_eight = reshaped[:, 1] + (47.0 / 20.0) * reshaped[:, 0] > (1497.0 / 4.0)
    relevant_ten = reshaped[:, 0] * -0.8 + reshaped[:, 1] * -1.3 + reshaped[:, 2] * .75 < -250
    relevant_eleven = reshaped[:, 0] * (61.0 / 105.0) + reshaped[:, 1] > (1164.0 / 5.0)


    relevant = np.logical_or(relevant, relevant_two)
    relevant = np.logical_or(relevant, relevant_four)
    relevant = np.logical_or(relevant, relevant_five)
    relevant = np.logical_or(relevant, relevant_six)
    relevant = np.logical_or(relevant, relevant_seven)
    relevant = np.logical_or(relevant, relevant_eight)
    relevant = np.logical_or(relevant, relevant_ten)
    relevant = np.logical_or(relevant, relevant_eleven)
    return relevant
