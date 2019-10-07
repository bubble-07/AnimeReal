
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#DOMAIN CONSTANTS
part_names = ["Neck", "Head", "BButton", 
              "L_Shoulder", "L_Elbow", "L_Hand",
              "L_Hip", "L_Knee", "L_Foot",
              "R_Shoulder", "R_Elbow", "R_Hand",
              "R_Hip", "R_Knee", "R_Foot"]

num_parts = len(part_names)

#Part sided-ness -- "L" for "left side", "R" for right side, "C" for centered
part_sides = ["C", "C", "C",
              "L", "L", "L",
              "L", "L", "L",
              "R", "R", "R",
              "R", "R", "R"]
#For each part, this stores the index analogous part on the other (left/right) side of the body
#or just the unmodified index if the part is centered
mirror_part = [0, 1, 2,
               9, 10, 11,
               12, 13, 14,
               3, 4, 5,
               6, 7, 8]


#TODO: Add PAFs back in?
'''
R_Upper_Arm = (R_Shoulder, R_Elbow)

L_Upper_Arm = (L_Shoulder, L_Elbow)

R_Lower_Arm = (R_Elbow, R_Wrist)

L_Lower_Arm = (L_Elbow, L_Wrist)

R_Upper_Leg = (R_Hip, R_Knee)

L_Upper_Leg = (L_Hip, L_Knee)

R_Lower_Leg = (R_Knee, R_Ankle)

L_Lower_Leg = (L_Knee, L_Ankle)

Chest_Nose = (Chest, Nose)

R_Nose_Eye = (Nose, R_Eye)

L_Nose_Eye = (Nose, L_Eye)

R_Eye_Ear = (R_Eye, R_Ear)

L_Eye_Ear = (L_Eye, L_Ear)

R_Side = (R_Shoulder, R_Hip)

L_Side = (L_Shoulder, L_Hip)

part_segments = [R_Upper_Arm, L_Upper_Arm, R_Lower_Arm, L_Lower_Arm,
                 R_Upper_Leg, L_Upper_Leg, R_Lower_Leg, L_Lower_Leg,
                 Chest_Nose, R_Nose_Eye, L_Nose_Eye, R_Eye_Ear,
                 L_Eye_Ear, R_Side, L_Side]

num_part_segments = len(part_segments)
'''

#The number of feature maps in each output of the neural net
num_field_maps = num_parts
