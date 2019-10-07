#A script which takes OpenImages v4 (https://storage.googleapis.com/openimages/web/download.html)
#images and bounding box .csvs and outputs image, label box segmentations
#into a destination tfrecord directory
import sys
import csv
import time
from PIL import Image
from io import BytesIO
import numpy as np
import cv2
import requests
from OpenImagesTFWriter import *

class_descriptions = { "Human eye" : "/m/014sv8",
                       "Human beard" : "/m/015h_t",
                       "Person" : "/m/01g317",
                       "Human mouth" : "/m/0283dt1",
                       "Human foot" : "/m/031n1",
                       "Human leg" : "/m/035r7c",
                       "Human ear" : "/m/039xj_",
                       "Human hair" : "/m/03q69",
                       "Human head" : "/m/04hgtk",
                       "Girl" : "/m/05r655",
                       "Human face" : "/m/0dzct",
                       "Human arm" : "/m/0dzft4",
                       "Human nose" : "/m/0k0pj",
                       "Human hand" : "/m/0k65p",
                       "Boy" : "/m/01bl7v",
                       "Woman" : "/m/03bt1vf",
                       "Man" : "/m/04yx4",
                       "Human body" : "/m/02p0tk3"}
rev_class_descriptions = {v: k for k, v in class_descriptions.iteritems()}


class_ids = class_descriptions.values()

full_body_class_ids = set([class_descriptions["Boy"], class_descriptions["Woman"],
                       class_descriptions["Man"], class_descriptions["Girl"],
                       class_descriptions["Person"], class_descriptions["Human body"]])

#Codes (position of the one bit per collection of classes) for the output label images
class_codes = { "Girl" : 0,
                "Boy" : 0,
                "Woman" : 0,
                "Man" : 0,
                "Person" : 0,
                "Human body" : 0,
                "Human eye" : 1,
                "Human beard" : 2,
                "Human mouth" : 3,
                "Human foot" : 4,
                "Human leg" : 5,
                "Human ear" : 6,
                "Human hair" : 7,
                "Human head" : 8,
                "Human face" : 9,
                "Human arm" : 10,
                "Human nose" : 11,
                "Human hand" : 12}

num_class_codes = 13

#Okay, great. Now we can actually do stuff. 
imagesCSV = sys.argv[1]
boxesCSV = sys.argv[2]
destRoot = sys.argv[3]

writer = OpenImagesTFWriter(destRoot)

imageIdsToBoxes = {}

imageRejectList = []


#Will contain only those boxes containing things in the values of class_descriptions
relevantBoxesRows = []
with open(boxesCSV) as csvfile:
    boxesReader = csv.reader(csvfile)
    for row in boxesReader:
        box_class_id = row[2]
        if (box_class_id not in class_ids):
            #We only care about our pre-defined list of classes
            continue
        box_image_id = row[0]
        box_is_group_of = int(row[-3])
        if (box_is_group_of > 0):
            imageRejectList.append(box_image_id)
            #Disregard groups of people with no discernable boundary -- we don't want those
            #In fact, throw out the whole image!
            continue
        box_coordinates = row[4:8]
        box_is_depiction = int(row[-2])
        if (box_is_depiction > 0):
            #We don't care about _depictions_ of humans and their parts
            continue
        our_box = (box_class_id, box_coordinates)
        if (box_image_id in imageIdsToBoxes):
            #Okay, great. There must be a pre-existing list of boxes there.
            #Append our box.
            imageIdsToBoxes[box_image_id].append(our_box)
        else:
            #Otherwise, create the list entry
            imageIdsToBoxes[box_image_id] = [our_box]

imageRejectSet = set(imageRejectList)

#Okay, great, now remove from consideration all images which don't happen to include
#a full-body annotation
full_body_class_ids
for imageId in imageIdsToBoxes.keys():
    if (imageId in imageRejectSet):
        del imageIdsToBoxes[imageId]
        continue
    box_list = imageIdsToBoxes[imageId]
    class_list = []
    for box in box_list:
        class_list.append(box[0])
    class_set = set(class_list)
    if (len(class_set.intersection(full_body_class_ids)) == 0):
        #Delete it from the dict!
        del imageIdsToBoxes[imageId]



total_images = len(imageIdsToBoxes)

remaining_images = total_images

#Okay, great. Now we have imageIdsToBoxes filled. Now, we go through
#the images csv file and whenever we find an image for which
#we have annotated boxes, we'll send it off to the TFRecord factory
with open(imagesCSV) as csvfile:
    imagesReader = csv.reader(csvfile)
    start = time.time()
    s = requests.Session()
    for row in imagesReader:
        imageId = row[0]
        if (imageId not in imageIdsToBoxes):
            continue

        print "Images remaining: ", remaining_images
        remaining_images -= 1
        processed_images = total_images - remaining_images
        images_per_second = processed_images / (time.time() - start)
        print "Images per second: ", images_per_second
        hours_left = remaining_images / images_per_second
        hours_left /= 60.0
        hours_left /= 60.0
        print "Time remaining (hrs): ", hours_left
        print ""

        url = row[-2]
        if (len(url) == 0):
            continue
        rotation = 0
        try:
            rotation = int(float(row[-1])) / 90
        except ValueError:
            rotation = 0
        #Great, now try to download the image from the passed url

        img_data_req = s.get(url)
        if (img_data_req.status_code != requests.codes.ok):
            print "Image not successfully downloaded! Code: ", img_data_req.status_code
            continue

        img_data = img_data_req.content


        img = Image.open(BytesIO(img_data))
        img = img.convert(mode="RGB")

        img = np.asarray(img)


        img = cv2.resize(img, (256, 256))

        #Okay, now apply the rotation we need to it
        img = np.rot90(img, k=rotation)

        #Okay, great, we have an image. Now, time to layer in bounding boxes!
        anno_masks = []
        for i in range(num_class_codes):
            anno_masks.append(np.zeros((256, 256), dtype=np.bool_))

        result_mask = np.zeros((256, 256), dtype=np.uint16)

        boxes = imageIdsToBoxes[imageId]
        for box_class, box_coordinates in boxes:
            class_name = rev_class_descriptions[box_class]
            class_code = class_codes[class_name]

            box_low_x = int(float(box_coordinates[0]) * 255.0)
            box_high_x = int(float(box_coordinates[1]) * 255.0)
            box_low_y = int(float(box_coordinates[2]) * 255.0)
            box_high_y = int(float(box_coordinates[3]) * 255.0)

            #Modify anno_masks[class_code] by a logical "or" with the bounding box mask
            anno_masks[class_code][box_low_y:box_high_y, box_low_x:box_high_x] = True
        for i in range(num_class_codes):
            class_numeral = 2 ** i
            result_mask += anno_masks[i].astype(np.uint16) * class_numeral

        #We have it encoded like we wanted! Write to the output .tfrecord dir
        writer.add(img, result_mask)


        '''
        #Okay, great, we have an image. Now, display the image
        cv2.imshow("Image", img)
        cv2.waitKey(1)
        '''
writer.flush() 


