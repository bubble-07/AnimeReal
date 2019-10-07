#An object responsible for managing multiple CloudManagers
#each of which corresponds to a different camera
#This merges together the capture timelines of all
#CloudManagers into one big chronological timeline,
#with the (hopefully correct!) spatial alignment
#If any of the CloudManagers yields frames in a non-chronological
#order, this will fail horribly, but if so, it's
#garbage-in-garbage-out. Similarly, the timestamps on
#all of the CloudManagers this thing is managing
#should already be synchronized.

import CloudManager

class MultiCloudManager():
    def __init__(self, cloudManagers):
        self.cloudManagers = cloudManagers
        self.origCloudManagers = cloudManagers
        self.cloudOverride = None

    def unsafeScrubUnderlying(self, num_frames):
        for cloudManager in self.cloudManagers:
            cloudManager.scrub(num_frames)
    def smolTimestampIndex(self):
        minTimestamp = float("inf")
        minInd = -1
        for i in range(len(self.cloudManagers)):
            timestamp = self.cloudManagers[i].getTimestamp()
            if (timestamp < minTimestamp):
                minInd = i
                minTimestamp = timestamp
        return minInd
    def getTimestamp(self):
        ind = self.smolTimestampIndex()
        return self.cloudManagers[ind].getTimestamp()
    def setCloudOverride(self, override):
        self.cloudOverride = override
    def getCloud(self):
        if (self.cloudOverride is not None):
            return self.cloudOverride
        indToGet = self.smolTimestampIndex()
        return self.cloudManagers[indToGet].getCloud()
    def getOrigFrame(self):
        indToGet = self.smolTimestampIndex()
        return self.cloudManagers[indToGet].getOrigFrame()
    #TODO: More sensible frame indices -- these ones now suck for annotations
    def getFrameIndices(self):
        indToGet = self.smolTimestampIndex()
        return self.cloudManagers[indToGet].getFrameIndices()
    def scrub(self, num_frames):
        #TODO: implement reversing!
        return self.advance()
    def advance(self):
        if (len(self.cloudManagers) == 0):
            return False
        indToAdvance = self.smolTimestampIndex()
        scrub_success = self.cloudManagers[indToAdvance].scrub(1)
        if (scrub_success):
            return True
        else:
            #It must be the case that we have an un-advanceable cloud manager!
            #remove it from the list of managers
            #Since there are not zero cloud managers, the other cloud managers
            #must still have frames, so deleting the one with the smallest
            #timestamp will perform an advancement
            if (len(self.cloudManagers) == 1):
                return False
            del self.cloudManagers[indToAdvance]
            return True
                

            


