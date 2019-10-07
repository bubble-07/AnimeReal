import numpy as np
import cv2
import msgpack
import scipy as sp
import ColorFilters
from scipy.spatial import cKDTree
import KinectTwoCamera

def bogusEmptyFrame():
    rgb = np.zeros((1080, 1920, 3), dtype=np.uint8)
    depth = np.zeros((424, 512), dtype=np.float32)
    timestamp = float("inf")
    return Frame(timestamp, rgb, depth)

class Frame:
    #A frame is really just an rgb image + a depth image + a timestamp,
    #but nothing more
    #more
    #TODO: also use camera identifier here
    def __init__(self, timestamp, rgb, depth):
        self.timestamp = timestamp
        self.rgb = rgb
        self.depth = depth
    def getTimestamp(self):
        return self.timestamp
    #Returns a 424x512 RGB image obtained using the
    #KinectTwo projection onto the depth index coordinates
    #(with black pixels)
    #and a boolean mask for all non-green pixels
    def getDepthRegisteredRGBAndMask(self, opening_iterations=0, closing_iterations=10, remask_closing_iterations=2):
        #First thing's first -- obtain the correct image slice
        depth_rgb = np.transpose(KinectTwoCamera.depth_buffer_to_rgb(self.getDepth()), axes=(1, 0, 2))
        rgb_inbounds = np.transpose(KinectTwoCamera.depth_buffer_to_rgb_inbounds(self.getDepth()))

        flat_rgb_inbounds_inds = np.where(rgb_inbounds.flatten())

        flat_depth_rgb = np.reshape(depth_rgb, (-1,2))
        flat_relevant_depth_rgb = flat_depth_rgb[flat_rgb_inbounds_inds]
        depth_relevant_rgb_y = flat_relevant_depth_rgb[:, 1]
        depth_relevant_rgb_x = flat_relevant_depth_rgb[:, 0]

        relevant_img_slice = self.getRGBForDisplay()[depth_relevant_rgb_y, depth_relevant_rgb_x, 0:3]
        flat_relevant_img_slice = np.reshape(relevant_img_slice, (-1, 3))

        flat_image_slice = np.reshape(np.zeros((424, 512, 3), dtype=np.uint8), (-1, 3))
        flat_image_slice[flat_rgb_inbounds_inds] = flat_relevant_img_slice


        greenMask = ColorFilters.pixelSpaceVectorMask(ColorFilters.nonTemplateMaskGreenScreen, flat_image_slice)

        tot_mask_flat = np.logical_and(rgb_inbounds.flatten(), greenMask)
        tot_mask = np.reshape(tot_mask_flat, (424, 512))

        #Okay, that's not all, though -- compute the binary opening followed by closing of the image!
        morphed_mask = np.copy(tot_mask)

        if (opening_iterations > 0):
            morphed_mask = sp.ndimage.morphology.binary_opening(morphed_mask, iterations=opening_iterations)
        if (closing_iterations > 0):
            morphed_mask = sp.ndimage.morphology.binary_closing(morphed_mask, iterations=closing_iterations)

        morphed_mask_flat = morphed_mask.flatten()



        #Okay, great. Now what we do is we take the mask above and use it to label the largest
        #connected image component
        structure = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]

        label_array, num_features = sp.ndimage.measurements.label(morphed_mask, structure=structure)
        flat_label_array = label_array.flatten()
        flat_nonzero_label_array = flat_label_array[np.nonzero(flat_label_array)]
        most_common_label, _ = sp.stats.mode(flat_nonzero_label_array)
        if (most_common_label.size < 1):
            #Largest component dne!
            #Give an empty image and an all-false map!
            image_result = np.zeros((424, 512, 3), dtype=np.uint8)
            mask_result = np.zeros((424, 512)) == 1
            return (image_result, mask_result)

        most_common_label = most_common_label[0]
        #Okay, great. Now, find all instances of the most common label, and make it into a mask
        common_mask_flat = flat_label_array == most_common_label

        complete_mask_flat = np.logical_and(tot_mask_flat, common_mask_flat)
        complete_mask = np.reshape(complete_mask_flat, (424, 512))

        if (remask_closing_iterations > 0):
            complete_mask = sp.ndimage.morphology.binary_closing(complete_mask, iterations=remask_closing_iterations)
            complete_mask_flat = complete_mask.flatten()


        #Set all non-mask pixels to green
        non_complete_mask_flat = np.logical_not(complete_mask_flat)
        flat_image_slice[non_complete_mask_flat] = np.array([0, 255, 0], dtype=np.uint8)


        #Okay, that was quite a bit of work to get the image slice, but we have it!
        image_slice = np.reshape(flat_image_slice, (424, 512, 3))

        return image_slice, complete_mask




    def getRGB(self):
        return self.rgb
    def getRGBForDisplay(self):
        result = self.rgb[:,:,0:3]
        result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
        return result
    def getDepth(self):
        return self.depth

    def maskOutColor(self, lowerHue=67, upperHue=95, lowerSat=140, upperSat=255,
            lowerValue=50, upperValue=255):
        opposite = self.maskInColor(lowerHue, upperHue, lowerSat, upperSat, lowerValue,
                                    upperValue)
        return 1.0 - opposite

    #Returns a binary mask which is '1' where there are blue pixels (by default)
    def maskInColor(self, lowerHue=95, upperHue=117, lowerSat=150, upperSat=255,
            lowerValue=50, upperValue=255):
        #Do everything in the HSV color space
        hsvImage = cv2.cvtColor(self.rgb, cv2.COLOR_BGR2HSV)
        lower=np.array([lowerHue, lowerSat, lowerValue])
        upper=np.array([upperHue, upperSat, upperValue])
        mask = cv2.inRange(hsvImage, lower, upper)
        return mask
    
    #Given an frame and another which represents a (presumably green-screen)
    #background, and a discard tolerance proximity in millimeters,
    #filter out the backgorund
    @staticmethod
    def filterBackgroundOnDepth(frame, background, tol=40.0):
        resultDepth = np.copy(frame.getDepth())
        #Discard depth points for which we're within depth tolerance of the background
        resultDepth[np.nonzero(abs(background.getDepth() - frame.getDepth()) < tol)] = 0.0
        return Frame(frame.timestamp, frame.getRGB(), resultDepth)

    #Applies a gaussian convolution to a depth frame
    @staticmethod
    def gaussianConvolveDepth(frame, k_neighbors=10, sigma=1.0):
        depth_unknown = frame.getDepth() < 1.0
        #Fill out the depth frame where values are unknown
        depth_filled = frame.getFillDepthNearestK(k=k_neighbors)
        #Run that through a gaussian filter
        depth_convolved = sp.ndimage.filters.gaussian_filter(depth_filled, sigma, mode='nearest')
        #Finally, set all positions that were unknown before back to being unknown
        depth_convolved[depth_unknown] = 0

        return Frame(frame.timestamp, frame.getRGB(), depth_convolved)

    #For a given frame, return a version of the depth values
    #where missing (0.0) depths are replaced with an average
    #of the k-nearest-neighbors' depth
    def getFillDepthNearestK(self, k=10):
        depth = np.copy(self.getDepth())
        #Build a kd tree of filled depth indices
        h, w = depth.shape
        filledPositions = []
        filledValues = []
        unfilledPositions = []
        for i in range(h):
            for j in range(w):
                if (depth[i, j] > 1.0):
                    filledValues.append(depth[i, j])
                    filledPositions.append([i, j])
                else:
                    unfilledPositions.append(np.array([i, j], dtype=np.float32))

        filledPositions = np.array(filledPositions, dtype=np.float32)
        filledValues = np.array(filledValues, dtype=np.float32)

        kdTree = cKDTree(filledPositions)

        for pos in unfilledPositions:
            _, neighborInds = kdTree.query(pos, k=k)
            neighborValues = filledValues[neighborInds]
            meanValue = np.mean(neighborValues, axis=0)
            i, j = pos
            i = int(i)
            j = int(j)
            depth[i][j] = meanValue
        return depth

        

    #Averages the rgb and depth values in a given frame
    @staticmethod
    def averageFrames(frames, lowDepth=1, minSamps=15):
        rgbs = [frame.getRGB() for frame in frames]
        depths = []
        for frame in frames:
            depth = np.copy(frame.getDepth())
            thresh_indices = depth < lowDepth
            depth[thresh_indices] = 0
            depths.append(depth)
        #Compute denominator for depth averaging, since there may be missing depth points!
        depthCountArray = sum([np.array(x >= 1.0, dtype=np.int32) for x in depths])
        #Prevent division-by-zero errors on depth averaging
        modCountArray = (depthCountArray == 0) + depthCountArray
        depthSum = sum(depths)
        depthAverage = np.divide(depthSum, modCountArray)

        underMinSamps = depthCountArray < minSamps
        depthAverage[underMinSamps] = 0

        print depthCountArray
        print "Closest average point", np.amin(depthAverage)
        print "Lowest count", np.amin(depthCountArray)

        rgbSum = 0.0
        for rgb in rgbs:
            rgbSum += np.array(rgb, dtype=np.float32)
        rgbAverage = rgbSum / float(len(frames))
        
        rgbAverage = np.array(rgbAverage, np.int32)

        return Frame(frames[0].getTimestamp(), rgbAverage, depthAverage)
