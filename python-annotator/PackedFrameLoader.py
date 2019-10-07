import numpy as np
import msgpack
from Frame import *

def buildStandardFrame(versionCode, deviceCode, timestamp, rgbArray, depthArray):
    #TODO: Actually perform standardization to deal with versions and devices here!
    #TODO: Loading may also need to be completely different if you're using compressed
    #data!
    return Frame(timestamp, rgbArray, depthArray)

def loadPackedFrames(cameraFolder, datFileName):
    depthFilePath = cameraFolder + "/depth/" + datFileName
    rgbFilePath = cameraFolder + "/rgb/" + datFileName

    timestamps = []
    depthArrays = []
    rgbArrays = []
    versionCode, deviceCode = (None, None)

    with open(depthFilePath, 'rb') as depthFile:
        unpacker = msgpack.Unpacker(depthFile, raw=False)
        on_header = True
        for unpacked in unpacker:
            if on_header:
                on_header = False
                (versionCode, deviceCode) = unpacked
            else:
                (timestamp, height, width, img_bytes) = unpacked
                loaded_img = np.reshape(np.frombuffer(img_bytes, dtype=np.float32), 
                                                (height, width))
                depthArrays.append(loaded_img)
                timestamps.append(timestamp)
        
    with open(rgbFilePath, 'rb') as rgbFile:
        unpacker = msgpack.Unpacker(rgbFile, raw=False)
        on_header = True
        for unpacked in unpacker:
            if on_header:
                on_header = False
            else:
                (_, height, width, chan, img_bytes) = unpacked
                loaded_img = np.reshape(np.frombuffer(img_bytes, dtype=np.uint8),
                                        (height, width, chan))
                rgbArrays.append(loaded_img) 
    #TODO: Error handling for when the parallel arrays are not of the same size!!!
    #(possibly due to early abort of writing process?)
    #Great, now, from those parallel arrays, build something
    frames = []
    for i in range(len(rgbArrays)):
        frames.append(buildStandardFrame(versionCode, deviceCode,
                                        timestamps[i], rgbArrays[i], depthArrays[i]))
    return frames

        

    
