import StandardBody
import skimage
import numpy as np
import scipy as sp
import cv2

#Given a point in template image coordinates, returns
#a point in standard body coordinates, together with the
#weighting vector for computing distances
#(used to switch between cylindrical and spherical distances)
def point_pos(x, y, z):
    x_t = StandardBody.xmin
    y_t = StandardBody.ymin
    z_t = StandardBody.zmin
     
    x_c = StandardBody.xspread / (297.0)
    y_c = StandardBody.yspread / (430.0)
    z_c = StandardBody.zspread / (82.0)

    return ([x_t + x_c * x, y_t + y_c * y, z_t + z_c * z], [1.0, 1.0, 1.0])

nose_z = 82.0
ear_z = 41.0
eye_z = 0.25 * ear_z + .75 * nose_z
shoulder_z = 35.0
elbow_z = 18.0
wrist_z = 56.0
hip_z = 41.0
ankle_z = 15.0
knee_z = 58.0


l_shoulder_pos = point_pos(99, 94, shoulder_z)
r_shoulder_pos = point_pos(188, 94, shoulder_z)
l_elbow_pos = point_pos(36, 97, elbow_z)
r_elbow_pos = point_pos(250, 98, elbow_z)
l_wrist_pos = point_pos(17, 56, wrist_z)
r_wrist_pos = point_pos(272, 52, wrist_z)
l_hip_pos = point_pos(101, 214, hip_z)
r_hip_pos = point_pos(191, 214, hip_z)
l_knee_pos = point_pos(74, 304, knee_z)
r_knee_pos = point_pos(216, 304, knee_z)
l_ankle_pos = point_pos(53, 415, ankle_z)
r_ankle_pos = point_pos(234, 408, ankle_z)
nose_pos = point_pos(144, 39, nose_z)
l_ear_pos = point_pos(121, 35, ear_z)
r_ear_pos = point_pos(167, 35, ear_z)
l_eye_pos = point_pos(132, 29, eye_z)
r_eye_pos = point_pos(157, 29, eye_z)

neck_base_pos = point_pos(143, 66, 44)

coco_keypoints = [nose_pos, l_eye_pos, r_eye_pos, 
             l_ear_pos, r_ear_pos, l_shoulder_pos, r_shoulder_pos,
             l_elbow_pos, r_elbow_pos, l_wrist_pos, r_wrist_pos,
             l_hip_pos, r_hip_pos, l_knee_pos, r_knee_pos,
             l_ankle_pos, r_ankle_pos]
coco_keypoint_positions = [ a for a,b in coco_keypoints ]
coco_keypoint_weights = [ b for a,b in coco_keypoints ]

num_coco_keypoints = len(coco_keypoints)

names = ["Nose", "L_Eye", "R_Eye", "L_Ear", "R_Ear",
         "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
         "L_Wrist", "R_Wrist", "L_Hip", "R_Hip", "L_Knee",
         "R_Knee", "L_Ankle", "R_Ankle"]

index_dict = {}
for i in range(len(names)):
    index_dict[names[i]] = i
index_dict["Neck_Base"] = len(names)

pseudopartNames = ["Neck_Base"]

all_keypoints = coco_keypoints + [neck_base_pos]
all_keypoint_positions = [ a for a,b in all_keypoints ]

allNames = names + pseudopartNames

#given a num_coco_keypointsx3 array of positions, 
#expand to a dictionary with pseudo-points included
def dict_expand_keypoint_array(keypoint_array):
    min_x = float('inf')
    max_x = float('-inf')
    min_y = float('inf')
    max_y = float('-inf')

    result = {}
    for i in range(keypoint_array.shape[0]):
        point = keypoint_array[i]
        x = point[0]
        y = point[1]
        v = point[2]
        if (v < 0.1):
            #Don't include low-confidence points in bounding-box computation
            continue

        result[names[i]] = point
        if (x > max_x):
            max_x = x
        if (x < min_x):
            min_x = x
        if (y > max_y):
            max_y = y
        if (y < min_y):
            min_y = y

    
    #Great, now include pseudo-points
    if (("L_Shoulder" in result) and ("R_Shoulder" in result)):
        neck_base_pos = (result["L_Shoulder"] + result["R_Shoulder"]) * 0.5
        result["Neck_Base"] = neck_base_pos

    boundingBox = [min_x, max_x, min_y, max_y]

    return result, boundingBox

lineConnected = [["L_Shoulder", "L_Elbow"],
                 ["R_Shoulder", "R_Elbow"],
                 ["L_Elbow", "L_Wrist"],
                 ["R_Elbow", "R_Wrist"],
                 ["R_Hip", "R_Knee"],
                 ["L_Hip", "L_Knee"],
                 ["L_Knee", "L_Ankle"],
                 ["R_Knee", "R_Ankle"],
                 ["Neck_Base", "Nose"]]

polyConnected = [["R_Shoulder", "L_Shoulder",
                  "L_Hip", "R_Hip"],
                 ["Nose", "L_Eye", "L_Ear"],
                 ["Nose", "R_Eye", "R_Ear"],
                 ["R_Eye", "L_Eye", "Nose"]]

polyWeight = 1
lineWeight = 8

allConnected = []
for poly in polyConnected:
    allConnected.append((poly, polyWeight))

for poly in polyConnected:
    for i in range(len(poly)):
        for j in range(i):
            start = poly[i]
            end = poly[j]
            newLine = [start, end]
            allConnected.append((newLine, polyWeight))

for line in lineConnected:
    allConnected.append((line, lineWeight))

def clamp(x, below, above):
    if (x < below):
        return below
    if (x > above):
        return above
    return x

def drawLine(rowCoords, colCoords, rect_shape):
    min_val = -5000
    max_val = 5000

    start_x = clamp(colCoords[0], min_val, max_val)
    end_x = clamp(colCoords[1], min_val, max_val)
    start_y = clamp(rowCoords[0], min_val, max_val)
    end_y = clamp(rowCoords[1], min_val, max_val)
    row_coords, col_coords = skimage.draw.line(start_y, start_x, end_y, end_x)

    high_x = col_coords >= rect_shape[1]
    low_x = col_coords < 0
    high_y = row_coords >= rect_shape[0]
    low_y = row_coords < 0

    out_of_bounds = np.logical_or(np.logical_or(high_x, low_x), np.logical_or(high_y, low_y))

    in_bounds = np.logical_not(out_of_bounds)

    row_coords = row_coords[in_bounds]
    col_coords = col_coords[in_bounds]

    return row_coords, col_coords

    


#Given a list of dictionaries containing 2d coordinates of points,
#and a shape tuple, draw in the keypoints to a tensor of template positions

def draw_keypoint_array(keypoint_dicts, rect_shape):
    result_template = np.zeros([rect_shape[0], rect_shape[1], 3], dtype=np.float32)
    result_nonzero = np.zeros([rect_shape[0], rect_shape[1], 1], dtype=np.uint8)
    
    for keypoint_dict in keypoint_dicts:
        #First, draw all of the polygon-connected points
        for polyPoints, polyWeight in allConnected:
            cornerCoordsList = []
            shouldDrawPoly = True
            for pointName in polyPoints:
                if (pointName not in keypoint_dict):
                    shouldDrawPoly = False
                    break
                else:
                    cornerCoordsList.append(keypoint_dict[pointName])

            if (not shouldDrawPoly):
                continue

            cornerCoordsArray = np.array(cornerCoordsList, dtype=np.int32)

            templatePosList = [all_keypoint_positions[index_dict[pointName]] for pointName in polyPoints]

            templatePosArray = np.array(templatePosList, dtype=np.float32)

            cornerRowCoords = cornerCoordsArray[:, 1]
            cornerColCoords = cornerCoordsArray[:, 0]

            rowCoords = cornerRowCoords
            colCoords = cornerColCoords

            #Get the row and column coords for each point to draw into the
            #result_template array
            if (len(polyPoints) > 2):
                interiorRowCoords, interiorColCoords = skimage.draw.polygon(rowCoords, colCoords, shape=rect_shape)
            else:
                interiorRowCoords, interiorColCoords = drawLine(rowCoords, colCoords, rect_shape)

            interiorCoordsArray = np.stack([interiorColCoords, interiorRowCoords], axis=-1)

            #Okay, great, now compute a distance matrix which yields the distances
            #from each interior point to each corner point
            corner_dist_mat = sp.spatial.distance_matrix(interiorCoordsArray, cornerCoordsArray)

            total_corner_dists = np.sum(corner_dist_mat, axis=-1)
            total_corner_dists = np.reshape(total_corner_dists, [-1, 1])

            corner_dist_complements = total_corner_dists - corner_dist_mat

            total_weights = np.sum(corner_dist_complements, axis=-1)
            total_weights = np.reshape(total_weights, [-1, 1])

            epsilon = 0.001
            total_weights += epsilon

            #This is an array which, for every interior point, it associates
            #it with the [0.0, 1.0] weights to place on the template coords
            #from each of the corners
            corner_weights = corner_dist_complements / total_weights

            corner_weights = np.clip(corner_weights, a_min=0.0, a_max=1.0)

            interiorTemplateCoords = np.matmul(corner_weights, templatePosArray)

            #Okay, great, now draw!
            result_template[interiorRowCoords, interiorColCoords] = interiorTemplateCoords
            result_nonzero[interiorRowCoords, interiorColCoords, :] = polyWeight

    return (result_template, result_nonzero)



















