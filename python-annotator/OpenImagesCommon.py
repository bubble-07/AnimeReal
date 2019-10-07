#Common constants for our dealings with the OpenImages dataset
#TODO: Move more stuff in here
import StandardBody

#Translates a bounding-box from the template image space
#to template real space
def bbox(x_min=0.0, x_max=297.0, y_min=0.0, y_max=430.0):
    
    x_t = StandardBody.xmin
    y_t = StandardBody.ymin
     
    x_c = StandardBody.xspread / (297.0)
    y_c = StandardBody.yspread / (430.0)

    result_xmax = x_max * x_c + x_t
    result_xmin = x_min * x_c + x_t
    result_ymin = y_min * y_c + y_t
    result_ymax = y_max * y_c + y_t
    return [result_xmin, result_xmax, result_ymin, result_ymax]

nose_bbox = bbox(135, 153, 18, 46)
ears_bbox = bbox(115, 172, 24, 50)
eyes_bbox = bbox(118, 167, 20, 45)
head_bbox = bbox(115, 172, 0, 67)
r_arm_bbox = bbox(0, 100, 55, 117)
l_arm_bbox = bbox(190, 297, 120, 50)
r_hand_bbox = bbox(0, 40, 0, 68)
l_hand_bbox = bbox(244, 297, 0, 64)
hair_bbox = bbox(87, 204, 0, 245)
leg_bbox = bbox(0, 297, 247, 388)
face_bbox = bbox(115, 172, 0, 67)
beard_bbox = bbox(119, 169, 46, 80)
mouth_bbox = bbox(115, 172, 46, 62)
foot_bbox = bbox(0, 297, 388, 430)







