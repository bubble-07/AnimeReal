import StandardBody
import numpy as np
#File for defining approximate body part filters on the template body
#Each one of these filters takes as input a collection of points in millimeters
#and filters them down to only those points which lie within the specified body
#part boundary

def pixelSpaceBodyMask(maskFunc, points):
    #First, convert the list of points into our
    #stupid imaginary standard-body pixel-derived coords

    x_t = StandardBody.xmin
    y_t = StandardBody.ymin
    p_t = np.array([x_t, y_t, 0.0])

    x_c = StandardBody.xspread / (296.0)
    y_c = StandardBody.yspread / (430.0)
    p_c = np.array([x_c, y_c, 1.0])

    standardPoints = (np.copy(points) - p_t) / p_c

    filteredInds = maskFunc(standardPoints)
    return filteredInds

def maskLeftInnerArm(reshaped):
    relevant = reshaped[:, 0] > 240
    relevant_two = reshaped[:, 0] < 220
    relevant_three = reshaped[:, 1] > 132

    relevant = np.logical_or(relevant, relevant_two)
    relevant = np.logical_or(relevant, relevant_three)

    return relevant

def maskRightInnerArm(reshaped):
    relevant = reshaped[:, 0] > 70
    relevant_two = reshaped[:, 0] < 50
    relevant_three = reshaped[:, 1] > 132

    relevant = np.logical_or(relevant, relevant_two)
    relevant = np.logical_or(relevant, relevant_three)

    return relevant
    

def maskLeftOuterArm(reshaped):
    relevant = reshaped[:, 1] > 75
    relevant_two = reshaped[:, 1] < 62
    relevant_three = reshaped[:, 0] < 245

    relevant = np.logical_or(relevant, relevant_two)
    relevant = np.logical_or(relevant, relevant_three)

    return relevant

def maskRightOuterArm(reshaped):
    relevant = reshaped[:, 1] > 75
    relevant_two = reshaped[:, 1] < 62
    relevant_three = reshaped[:, 0] > 40

    relevant = np.logical_or(relevant, relevant_two)
    relevant = np.logical_or(relevant, relevant_three)

    return relevant


def maskLeftUpperLeg(reshaped):
    relevant = reshaped[:, 0] > 142
    relevant_two = reshaped[:, 1] < 290
    relevant_three = reshaped[:, 1] > 310

    relevant = np.logical_or(relevant, relevant_two)
    relevant = np.logical_or(relevant, relevant_three)

    return relevant

def maskRightUpperLeg(reshaped):
    relevant = reshaped[:, 0] < 142
    relevant_two = reshaped[:, 1] < 290
    relevant_three = reshaped[:, 1] > 310

    relevant = np.logical_or(relevant, relevant_two)
    relevant = np.logical_or(relevant, relevant_three)

    return relevant

def maskLeftLowerLeg(reshaped):
    relevant = reshaped[:, 0] > 142
    relevant_two = reshaped[:, 1] > 390
    relevant_three = reshaped[:, 1] < 350

    relevant = np.logical_or(relevant, relevant_two)
    relevant = np.logical_or(relevant, relevant_three)
    return relevant


def maskRightLowerLeg(reshaped):
    relevant = reshaped[:, 0] < 142
    relevant_two = reshaped[:, 1] > 390
    relevant_three = reshaped[:, 1] < 350

    relevant = np.logical_or(relevant, relevant_two)
    relevant = np.logical_or(relevant, relevant_three)

    return relevant

def maskHead(reshaped):
    relevant = reshaped[:, 0] < 100
    relevant_two = reshaped[:, 0] > 200
    relevant_three = reshaped[:, 1] > 40

    relevant = np.logical_or(relevant, relevant_two)
    relevant = np.logical_or(relevant, relevant_three)

    return relevant

def maskTorso(reshaped):

    relevant = reshaped[:, 0] < 112.0
    relevant_two = reshaped[:, 0] > 182.0
    relevant_three = reshaped[:, 1] < 118.0
    relevant_four = reshaped[:, 1] > 212.0

    relevant = np.logical_or(relevant, relevant_two)
    relevant = np.logical_or(relevant, relevant_three)
    relevant = np.logical_or(relevant, relevant_four)

    return relevant




