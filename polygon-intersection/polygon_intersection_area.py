
import matplotlib.pyplot as plt
import numpy as np
from polygon import Polygon
from convex_intersect import convexIntersect

def get_intersection(poly1, poly2):
    # print (poly2.convex())
    # print (poly2.simple())
    # print (poly1.intersect(poly2))

    intersect = convexIntersect(poly1, poly2)
    # print (intersect)

    x, y = np.array(intersect.getContourPoints()).T
    x1, y1 = np.array(poly1.getContourPoints()).T
    x2, y2 = np.array(poly2.getContourPoints()).T
    plt.plot(x1, y1)
    plt.plot(x2, y2)
    plt.plot(x, y, c='k')
    for c in intersect.getPoints():
        plt.scatter(c[0], c[1])
    plt.gca().invert_yaxis()
    plt.show()
    return intersect

def getScore(outputPts, expectedPts, width, height):
    print ('output points:', outputPts)
    print ('expected points:', expectedPts)
    wholeArea = width * height
    outputPoly = Polygon(outputPts)
    expectedPoly = Polygon(expectedPts)
    intersectPoly = get_intersection(outputPoly, expectedPoly)
    #print("whole area", wholeArea)
    #print("polexpectedPoly area", outputPoly.getArea())
    #print("expectedPoly area", expectedPoly.getArea())
    #print("intersectPoly area", intersectPoly.getArea())
    precision = intersectPoly.getArea() / outputPoly.getArea() # True positive / (True Positive + false positive)
    recall =  intersectPoly.getArea()  / expectedPoly.getArea() # True positive / (True Positive + false negative)
    f1 = 2 * precision * recall / (precision + recall)
    print ("precision: ", precision * 100, "%")
    print ("recall: ", recall * 100, "%")
    print("f1 score: ", f1)
    return f1


def test_intersection1():
    poly1 = Polygon([[451.8107, 173.44763],
                     [531.7912, 173.45604],
                     [533.50757, 222.04575],
                     [453.52707, 222.03732]])

    poly2 = Polygon([[449.92476, 172.5057],
                     [530.552293, 172.5057],
                     [530.552293, 220.237058],
                     [449.92476, 220.237058]])
    get_intersection(poly1, poly2)

def test_included():
    poly1 = Polygon([[100, 100],
                     [100, 200],
                     [200, 200],
                     [200, 100]])

    poly2 = Polygon([[50, 50],
                     [50, 250],
                     [250, 250],
                     [250, 50]])
    get_intersection(poly1, poly2)

def test_inorder(): # not working
    poly1 = Polygon([[100, 100],
                     [100, 200],
                     [200, 100],
                     [200, 200]
                     ])

    poly2 = Polygon([[250, 50],
                     [50, 50],
                     [50, 250],
                     [250, 250]])
    intersection = get_intersection(poly1, poly2)
    print("poly1 area", poly1.getArea())
    print("poly2 area", poly2.getArea())
    print("intersection area", intersection.getArea(), 100 * intersection.getArea() / poly1.getArea(), "%")

def test_wb1(): # must be in order
    poly1 = Polygon([[57.0, 134.0], [1910.0, 144.0], [57.0, 969.0], [1915.0, 998.0] ])

    poly2 = Polygon([[40.0,120.0],[1920.0,120.0],[40.0,1001.0], [1920.0,1001.0]])
    intersection = get_intersection(poly1, poly2)
    print ("poly1 area", poly1.getArea())
    print ("poly2 area", poly2.getArea())
    print("intersection area", intersection.getArea(), 100 * intersection.getArea() / poly1.getArea(), "%")

def test_getScore1():
    expectedPts = [[57.0, 134.0], [1910.0, 144.0], [57.0, 969.0], [1915.0, 998.0]]
    outputPts = [[40.0, 120.0], [1920.0, 120.0], [40.0, 1001.0], [1920.0, 1001.0]]
    width = 1920
    height = 1080
    f1 = getScore(outputPts, expectedPts, width, height)

def test_getScore2():
    expectedPts = [[208,57],[986,141],[157,669],[965,693]]
    outputPts = [[0.0,0.0],[809.0,63.47858472998138],[0.0,630.0],[809.0,618.8025477707006]]
    width = 1280
    height = 720
    f1 = getScore(outputPts, expectedPts, width, height)

test_getScore1()
test_getScore2()
