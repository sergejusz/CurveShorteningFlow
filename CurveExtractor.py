import cv2
import numpy as np
import geometry as geom


class CurveExtractor():
    """
    Very simple class to extract curve from image
    following thinned curve.
    """
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
        """
        Moves current position by given direction code.
        :param directionCode: direction code 0..7.
        :param row: row of current position.
        :param col: column of current position.
        :return new position.
        """
        dy, dx = self.displacements[directionCode]
        return (row + dy, col + dx)

    def getDirections(self, directionCode):
        """
        Returns the list of directions for given direction code.
        Directions are selected in the way to keep given direction.
        :param directionCode: direction code 0..7.
        :return list of directions.
        """
        if directionCode >= 0 and directionCode <= 7:
            return self.lookupTable[directionCode]
        return []
       
    def loadFromFile(self, path):
        """
        Returns cv2 image from file.
        :param path: image file path.
        :return image object.
        """
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            if self.debugPrint: print("CurveExtractor.loadFromFile failed for '", path, "'")
        return image

    # find and collect closed curve points on image
    # curve should be closed and without intersections
    def extract(self, image, signalColor):
        """
        Extracts closed curve from given image.
        It is very primitive implementation of curve following on image.
        There should be no fork points on curve.
        :param image: image to extract curve from.
        :param signalColor: color of curve points.
        :return curve.
        """
        if image is None:
            if self.debugPrint: print("CurveExtractor.extract no valid image is supplied.")
            return geom.get_empty_curve()
            
        rows,cols = image.shape[:2]

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
            return geom.get_empty_curve()

        x = []
        y = []
        row = pos[1]
        col = pos[0]
        x.append(col)
        y.append(row)
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
                    x.append(col)
                    y.append(row)
                    image[row, col] = 0
                    nextDirection = d
                    if self.debugPrint: 
                        print("CurveExtractor.extract: directionCode=", d, " x=", col, " y=", row)
                    break
            if nextDirection == None:
                finished = True
            else:
                directionCode = nextDirection
        return np.array([x, y])

    def clearNeighborhood(self, image, x, y, w, h):
        """
        Fills rectangular neighborhood at given position with pixels of zero value.
        :param image: image object.
        :param x: x coordinate of position.
        :param y: y coordinate of position.
        :param w: width of neighborhood rectangle.
        :param h: height of neighborhood rectangle.
        """
        rows,cols = image.shape
        w1 = w // 2
        h1 = h // 2
        r1 = y - h1
        c1 = x - w1
        r = r1
        for i in range(h):
            c = c1
            for j in range(w):
                if r>=0 and r<rows and c>=0 and c<cols:
                    image[r+i, c+j] = 0


    # move along the curve and clear curve pixels on image
    def clearByCurve(self, image, curve, w, h):
        """
        Fills rectangular neighborhood at each point of given curve with pixels of zero value.
        :param image: image object.
        :param curve: curve points.
        :param w: width of neighborhood rectangle.
        :param h: height of neighborhood rectangle.
        """
        if image is None:
            if self.debugPrint: print("CurveExtractor.extract no valid image")
            return
        
        rows,cols = image.shape[:2]
        for i in range(len(curve)):
            self.clearNeighborhood(image, curve[0][i], curve[1][i], w, h)
            
            
