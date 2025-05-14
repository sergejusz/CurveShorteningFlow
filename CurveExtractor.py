import cv2

class CurveExtractor():
    
    def __init__(self):
        self.displacements = {
            0 : (-1, 0), 
            1 : (-1, 1),
            2 : (0, 1),
            3 : (1, 1),
            4 : (1, 0),
            5 : (1, -1),
            6 : (0, -1),
            7 : (-1, -1)}
            
        self.lookupTable = {
            0: [0,7,1,6,2,5,4,3],
            1: [1,0,2,7,3,6,4,5],
            2: [2,1,3,0,4,7,5,6],
            3: [3,2,4,1,5,0,6,7],
            4: [4,5,3,6,2,7,1,0],
            5: [5,4,6,3,7,0,2,1],
            6: [6,5,7,4,0,1,3,2],
            7: [7,6,0,5,1,2,4,3]
        }
        self.debugPrint = False


    def setDebug(debugFlag):
        self.debugPrint = debugFlag
        
    def getNextPosition(self, directionCode, row, col):
        displacement = self.displacements[directionCode]
        return (row + displacement[0], col + displacement[1])

    def getDirections(self, directionCode):
        if directionCode >= 0 and directionCode <= 7:
            return self.lookupTable[directionCode]
        return []
       
    def loadFromFile(self, path):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            if self.debugPrint: print("CurveExtractor.loadFromFile failed for '", path, "'")
        return image


    # find and collect closed curve points on image
    # curve should be closed and without intersections
    def extract(self, image, signalColor):
        if image is None:
            if self.debugPrint: print("CurveExtractor.extract no valid image is supplied.")
            return []
            
        rows,cols = image.shape

        # find curve point
        pos = None
        for row in range(rows):
            if pos != None:
                break
            for col in range(cols):
                k = image[row, col]
                if image[row, col] == signalColor:
                    pos = (col, row)

        if pos == None:
            return []

        curve = []
        row = pos[1]
        col = pos[0]
        curve.append((col, row))
        image[row, col] = 0
    
        finished = False
        directionCode = 0
        while not finished:
            directionCodes = self.getDirections(directionCode)
            nextDirection = None
            for d in directionCodes:
                nextPosition = self.getNextPosition(d, row, col)
                if image[nextPosition[0], nextPosition[1]] == signalColor:
                    row = nextPosition[0]
                    col = nextPosition[1]
                    curve.append((col, row))
                    image[row, col] = 0
                    nextDirection = d
                    if self.debugPrint: 
                        print("CurveExtractor.extract: directionCode=", d, " x=", col, " y=", row)
                    break
            if nextDirection == None:
                finished = True
            else:
                directionCode = nextDirection
        return curve

    def clearNeighborhood(self, image, x, y, w, h):
        rows,cols = image.shape
        w1 = w // 2
        h1 = h // 2
        r1 = y - h1
        c1 = x - w1
        r = r1
        for i in range(0, h):
            c = c1
            for j in range(0, w):
                if r>=0 and r<rows and c>=0 and c<cols:
                    image[r+i, c+j] = 0


    # move along the curve and clear curve pixels on image
    def clearByCurve(self, image, curve, w, h):
        if image is None:
            if self.debugPrint: print("CurveExtractor.extract no valid image")
            return
        
        if len(curve) == 0:
            if self.debugPrint: print("CurveExtractor.clearByCurve curve is empty")
            return
    
        rows,cols = image.shape
        for p in curve:
            self.clearNeighborhood(image, p[0], p[1], w, h)
            
            
