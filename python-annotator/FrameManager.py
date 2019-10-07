#An object responsible for loading frames from a given directory,
#and making them accessible in a sequential manner
import PackedFrameLoader
import Frame

class FrameManager():
    def __init__(self, sequenceRoot):
        self.sequenceRoot = sequenceRoot
        self.currentFileNumber = 0
        self.currentFrameNumber = 0
        self.currentFrames = []
        self.loadFramesFile()
    def loadFramesFile(self):
        filename = str(self.currentFileNumber) + ".dat"
        self.currentFrames = PackedFrameLoader.loadPackedFrames(self.sequenceRoot, filename)
    def seekTo(self, fileNumber, frameNumber):
        self.currentFileNumber = fileNumber
        self.loadFramesFile()
        self.currentFrameNumber = frameNumber
    #Returns all frames as a big, long array. Calling this is perhaps not a good idea
    def getAllFrames(self):
        seekFile = self.currentFileNumber
        seekFrame = self.currentFrameNumber
        self.currentFileNumber = 0
        self.currentFrameNumber = 0
        self.loadFramesFile()
        result = []
        while True:
            result.append(self.getFrame())
            success = self.tryAdvance(1) 
            if (not success):
                break
        self.seekTo(seekFile, seekFrame)
        return result
    def getFrameIndices(self):
        return (self.currentFileNumber, self.currentFrameNumber)
    def getFrame(self):
        #If there are no frames, return a bogus empty frame
        if (len(self.currentFrames) == 0):
            return Frame.bogusEmptyFrame()
        return self.currentFrames[self.currentFrameNumber]
    def getViewTime(self):
        return self.getFrame().getTimestamp()
    def tryAdvance(self, num_frames):
        print self.currentFileNumber, self.currentFrameNumber
        if (self.currentFrameNumber + num_frames < len(self.currentFrames)):
            self.currentFrameNumber += num_frames
        else:
            num_frames -= len(self.currentFrames) - self.currentFrameNumber - 1
            self.currentFrameNumber = 0
            self.currentFileNumber += 1
            try:
                self.loadFramesFile()
            except IOError:
                self.currentFrameNumber = len(self.currentFrames) - 1
                self.currentFileNumber -= 1
                self.loadFramesFile()
                return False
            return self.tryAdvance(num_frames)
        return True
    def tryRewind(self, num_frames):
        print self.currentFileNumber, self.currentFrameNumber
        if (self.currentFrameNumber - num_frames >= 0):
            self.currentFrameNumber -= num_frames
        else:
            num_frames -= self.currentFrameNumber + 1
            self.currentFileNumber -= 1
            try:
                self.loadFramesFile()
                self.currentFrameNumber = len(self.currentFrames) - 1
            except IOError:
                self.currentFrameNumber = 0
                self.currentFileNumber = 0
                self.loadFramesFile()
                return False
            return self.tryRewind(num_frames)
        return True
