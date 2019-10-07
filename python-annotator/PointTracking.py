import colorsys
import numpy as np
import ColorFilters
import math
#Collection of utilities for tracking
#points identified by colored squares on the
#super suit

def filterCloudToPoints(pointcloud):
    pointcloud.applyBackgroundFilter()
    pointcloud.applyColorFilter(ColorFilters.maskTooSaturated, negated=True)
    pointcloud.applyColorFilter(ColorFilters.maskRemnantYellow, negated=False)
    pointcloud.applyColorFilter(ColorFilters.maskRemnantWhite, negated=False)
    pointcloud.applyColorFilter(ColorFilters.maskRemnantRemnantWhite, negated=False)
    pointcloud.applyColorFilter(ColorFilters.maskRemnantRemnantRemnantWhite, negated=False)
    pointcloud.applyColorFilter(ColorFilters.maskRemnantRemnantYellow, negated=False)
    pointcloud.applyColorFilter(ColorFilters.maskRemnantRemnantRemnantYellow, negated=False)

#Definitions table drawing correspondences between
#point locations and colors in RGB. This is a big-ass table

colorPointCorrespondences = [
([[238, 171, 188], [254, 186, 197], [229, 200, 160]], [-479, -783, 99], "RedHand"),
([[252, 253, 213], [255, 247, 207], [255, 255, 217]], [-498, -780, 74], "RedHand"),
([[190, 208, 244], [184, 209, 249], [198, 213, 252]], [-530, -769, 61], "RedHand"),
([[255, 255, 209], [252, 255, 200], [248, 255, 203]], [-520, -664, 91], "RedHand"),
([[184, 198, 245], [189, 199, 250], [170, 192, 239]], [-558, -713, 57], "RedHand"),
([[255, 255, 157], [255, 255, 144], [255, 255, 146]], [-463, -541, -13], "RedArm"),
([[201, 186, 243], [197, 184, 240], [192, 181, 241]], [-322, -422, -17], "RedArm"),
([[248, 209, 214], [253, 221, 224], [245, 202, 212]], [17, -775, 150], "Torso"),
([[169, 209, 255], [158, 198, 250], [156, 196, 248]], [31, -694, 135], "Torso"),
([[202, 156, 231], [199, 153, 226], [206, 159, 231]], [-39, -682, 148], "Torso"),
([[252, 223, 228], [255, 216, 241], [255, 219, 240]], [-106, -459, 96], "Torso"),
([[168, 255, 213], [173, 255, 228], [169, 255, 220]], [71, -212, 155], "Torso"),
([[125, 202, 248], [137, 211, 250], [129, 207, 243]], [-163, -21, 55], "Torso"),
([[154, 250, 239], [174, 255, 254], [150, 251, 237]], [199, 30, 78], "Torso"),
([[242, 166, 241], [242, 169, 242], [235, 165, 235]], [-275, 261, 122], "WhiteLeg"),
([[228, 134, 196], [228, 134, 196], [228, 134, 194]], [-347, 479, 23], "WhiteLeg"),
([[110, 168, 216], [112, 171, 211], [115, 174, 218]], [-331, 649, -18], "WhiteLeg"),
([[233, 140, 169], [232, 142, 170], [214, 154, 179]], [-435, 815, 92], "WhiteLeg"),
([[248, 194, 230], [248, 192, 231], [245, 195, 232]], [226, 185, 122], "GreenLeg"),
([[215, 162, 130], [216, 179, 153], [222, 156, 132]], [232, 389, 34], "GreenLeg"),
([[217, 145, 209], [226, 140, 215], [225, 143, 215]], [358, 728, -34], "GreenLeg"),
([[231, 185, 135], [254, 217, 175], [239, 196, 141]], [335, 804, 9], "GreenLeg"),
([[221, 118, 165], [223, 115, 165], [239, 138, 182]], [391, 819, 76], "GreenLeg"),
([[237, 166, 162], [238, 166, 152], [240, 162, 160]], [334, -415, -7], "YellowArm"),
([[166, 255, 254], [137, 255, 255], [159, 255, 249]], [409, -449, -15], "YellowArm"),
([[141, 155, 138], [144, 144, 118], [146, 147, 142]], [453, -566, 21], "YellowArm"),
([[191, 255, 212], [189, 255, 209], [189, 255, 216]], [518, -642, 119], "YellowHand"),
([[252, 228, 158], [251, 233, 159], [247, 220, 141]], [438, -647, 117], "YellowHand"),
([[255, 255, 211], [255, 255, 190], [255, 255, 210]], [427, -678, 109], "YellowHand"),
([[253, 201, 239], [251, 212, 239], [250, 204, 233]], [465, -763, 125], "YellowHand"),
([[184, 192, 239], [192, 190, 237], [183, 189, 237]], [510, -795, 73], "YellowHand"),
([[200, 243, 224], [182, 249, 232], [196, 247, 228]], [546, -786, 99], "YellowHand"),
([[234, 207, 186], [255, 235, 217], [255, 233, 213]], [591, -748, 67], "YellowHand"),
([[253, 255, 255], [255, 255, 227], [251, 255, 234]], [497, -643, 79], "YellowHand"),
([[250, 224, 225], [254, 223, 231], [255, 225, 233]], [571, -728, 78], "YellowHand"),
([[179, 255, 246], [204, 254, 243], [196, 255, 245]], [539, -754, 109], "YellowHand"),
([[244, 201, 246], [222, 178, 237], [245, 203, 251]], [509, -773, 106], "YellowHand"),
([[246, 177, 244], [244, 177, 246], [232, 149, 231]], [462, -751, 104], "YellowHand"),
([[236, 227, 148], [243, 222, 159], [243, 221, 161]], [484, -493, -56], "YellowArm"),
([[255, 255, 232], [255, 255, 223], [253, 255, 215]], [-33, -777, -65], "Torso"),
([[255, 248, 229], [255, 249, 223], [255, 250, 224]], [-10, -614, -58], "Torso"),
([[235, 215, 250], [233, 219, 255], [230, 210, 245]], [-5, -290, -83], "Torso"),
([[221, 144, 226], [220, 165, 255], [216, 151, 235]], [143, 76, -89], "Torso"),
([[254, 255, 205], [246, 255, 176], [251, 255, 229]], [-90, 50, -118], "Torso"),
([[182, 204, 253], [199, 222, 255], [167, 195, 243]], [291, 345, 21], "GreenLeg"),
([[185, 255, 255], [206, 255, 255], [192, 255, 255]], [-438, -489, -110], "RedArm"),
([[145, 211, 246], [136, 212, 248], [138, 206, 241]], [-509, -631, 28], "RedArm"),
([[255, 255, 209], [254, 255, 181], [254, 255, 181]], [-470, -654, 77], "RedHand"),
([[202, 255, 241], [201, 255, 244], [202, 255, 245]], [-529, -685, 36], "RedHand"),
([[255, 204, 255], [248, 183, 251], [248, 186, 251]], [-504, -781, 72], "RedHand"),
([[150, 217, 246], [163, 224, 253], [150, 216, 248]], [-539, -756, 51], "RedHand"),
([[213, 201, 249], [209, 197, 245], [198, 190, 239]], [-381, -767, 34], "RedHand"),
([[161, 255, 253], [176, 255, 254], [177, 255, 255]], [-171, -102, -25], "Torso"),
([[255, 255, 231], [255, 255, 229], [255, 255, 228]], [-116, -730, 11], "Torso"),
([[244, 203, 147], [245, 207, 144], [245, 204, 160]], [-353, 492, -38], "WhiteLeg"),
([[167, 240, 249], [157, 237, 248], [172, 240, 251]], [-339, 370, 64], "WhiteLeg"),
([[214, 174, 215], [240, 175, 215], [235, 173, 210]], [-287, 274, 118], "WhiteLeg"),
([[223, 160, 127], [205, 149, 116], [206, 154, 117]], [238, 377, 85], "GreenLeg"),
([[81, 217, 203], [94, 210, 199], [78, 210, 196]], [340, 771, -131], "GreenLeg"),
([[255, 233, 255], [253, 234, 255], [255, 238, 255]], [-484, -461, -75], "RedArm"),
([[255, 254, 243], [255, 255, 240], [255, 255, 240]], [-498, -584, 26], "RedArm"),
([[254, 254, 254], [254, 254, 255], [255, 253, 254]], [133, -765, 48], "Torso"),
([[197, 199, 250], [194, 200, 248], [195, 205, 255]], [185, -102, -24], "Torso"), 
([[213, 117, 240], [197, 111, 234], [181, 111, 225]], [106, 63, -109], "Torso"),
([[252, 207, 250], [253, 212, 252], [250, 216, 253]], [355, 505, 20], "GreenLeg"),
([[139, 168, 246], [131, 168, 239], [132, 167, 248]], [316, 362, 20], "GreenLeg")]

huePositionPartList = []
pointList = []
for colors, point, part in colorPointCorrespondences:
    pointList.append(point)
    hues = []
    for color in colors:
        r, g, b = [None, None, None]
        try:
            r, g, b = color
        except ValueError:
            print color
        h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
        h = h * 360.0 
        hues.append(h)
    averageSin = 0
    averageCos = 0
    for hue in hues:
        sine = math.sin(math.radians(hue))
        cosine = math.cos(math.radians(hue))
        averageSin += sine
        averageCos += cosine
    averageSin /= 1.0 * len(hues)
    averageCos /= 1.0 * len(hues)
    averageRadianHue = math.atan2(averageSin, averageCos) % (math.pi * 2.0)
    averageDegreeHue = math.degrees(averageRadianHue)
    averageHue = averageDegreeHue / 2.0
    huePositionPartList.append((averageHue, point, part))
      
partHuePositionDict = {"GreenLeg" : {}, "YellowArm" : {}, "RedArm" : {}, "RedHand" : {},
                       "YellowHand" : {}, "WhiteLeg" : {}, "Torso" : {}}

for hue, position, part in huePositionPartList:
    partHuePositionDict[part][hue] = position

def getAllPointPositions():
    return np.array(pointList, dtype=np.float32)

#Given the hue of one of the markers, and the part it most likely resides in, 
#return the closest template position for the marker
def findTemplatePosition(markerHue, markerPart, giveUpHueThresh=40.0):
    #print "Part: ", markerPart
    #print "Marker hue: ", markerHue
    huePositionDict = partHuePositionDict[markerPart]
    #Okay, now that we've narrowed down the part, find the marker hue which
    #best fits 
    minHueDiff = 180.0
    bestHue = 0.0
    for hue in huePositionDict.keys():
    #    print "Hues: ", hue
        hueDiff = min(math.fabs(markerHue - hue), markerHue + 180.0 - hue, hue + 180.0 - markerHue)
        if (hueDiff < minHueDiff):
            bestHue = hue
            minHueDiff = hueDiff
    if (minHueDiff > giveUpHueThresh):
        return None
    #Return value in centimeter coordinates
    return np.array(huePositionDict[bestHue], dtype=np.float32) / 10.0
