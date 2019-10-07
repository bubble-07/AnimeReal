import numpy as np

#Point, together with a unique identifier
glob_identifier = 0
class IdentifiedPoint():
    def __init__(self, point):
        global glob_identifier
        self.id = glob_identifier
        glob_identifier += 1
        self.point = point
    def getIdentifier(self):
        return self.id
    def getPoint(self):
        return self.point
    def hasSameId(self, other):
        return self.id == other.id
    #Create a point which has the same identifier,
    #but where the point is different
    def createLinked(self, newPoint):
        #Do this garbage so we don't bump up the number of ids
        global glob_identifier
        result = IdentifiedPoint(newPoint)
        glob_identifier -= 1

        result.id = self.id
        return result
    #Derive a point which has the same identifier,
    #but where the point is transformed by the given
    #transformation function
    def deriveLinked(self, transFunc):
        return self.createLinked(transFunc(self.getPoint()))

class IdentifiedPointDict():
    def __init__(self):
        self.table = {}
    def add(self, item):
        identifier = item.getIdentifier()
        self.table[identifier] = item
    def get(self, item):
        identifier = item.getIdentifier()
        return self.table[identifier]
    def pop(self, item):
        identifier = item.getIdentifier()
        try:
            return self.table.pop(identifier)
        except KeyError:
            return None
    def getValues(self):
        return self.table.values()
    #Find the closest point to a given position,
    #and return it
    def getClosest(self, pos, thresh):
        closestItem = None
        closestDistance = float('+inf')
        for item in self.table.values():
            testPoint = item.getPoint()
            distance = np.linalg.norm(pos - testPoint)
            if (distance < closestDistance):
               closestDistance = distance
               closestItem = item
        if (closestDistance < thresh):
            return closestItem
        return None
            

     
