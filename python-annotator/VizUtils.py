import cv2
import numpy as np
from matplotlib import pyplot as plt

def displayHistogram(windowName, valueList):
    plt.hist(valueList)
    plt.title(windowName)
    plt.show()

def quickDisplayOrtho(windowName, pointList, colorList, NUM_POINTS=5000):
    N = pointList.shape[0]
    if (N < NUM_POINTS):
        displayOrtho(windowName, pointList, colorList)
    else:
        indices = np.random.choice(N, NUM_POINTS, replace=False)
        newPointList = pointList[indices]
        newColorList = colorList[indices]
        displayOrtho(windowName, newPointList, newColorList)


#Mostly for debugging, this takes a name for a cv2 namedwindow
#a list of points, and a list of colors, and pops up an orthographic-projected
#display of those points
def displayOrtho(windowName, pointList, colorList):
    #First, find the minimum and maximum values for x and y coordinates
    sorted_xes = np.sort(np.copy(pointList[:, 0]))
    sorted_ys = np.sort(np.copy(pointList[:, 1]))

    N = sorted_xes.shape[0]

    if (N < 2):
        return

    x_min = sorted_xes[0]
    x_max = sorted_xes[N - 1]

    y_min = sorted_ys[0]
    y_max = sorted_ys[N - 1]

    xres = 512
    yres = 424

    min_x_diff = (x_max - x_min) / float(xres)
    min_y_diff = (y_max - y_min) / float(yres)


    #Compute output positions
    positions = pointList[:, 0:2] - np.array([[x_min, y_min]], dtype=np.float32)
    positions = positions / np.array([[min_x_diff, min_y_diff]], dtype=np.float32)

    #Create an empty image
    image = np.zeros((yres + 1, xres + 1, 3), dtype=np.uint8)

    #Now, fill the image
    for i in range(N):
        x, y = positions[i]
        r, g, b = colorList[i, 0:3]
        color = np.array([b, g, r])
        image[int(y), int(x)] = color
    #Finally, display the image
    cv2.imshow(windowName, image)
    cv2.waitKey(10)


