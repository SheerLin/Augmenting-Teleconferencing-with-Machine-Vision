import cv2
import numpy as np
import os
import math
from reshape import four_point_transform

def emptyDirectory(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)

def crop_minAreaRect(img, rect):

    # rotate img
    angle = rect[2]
    rows,cols = img.shape[0], img.shape[1]
    M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
    img_rot = cv2.warpAffine(img,M,(cols,rows))

    # rotate bounding box
    rect0 = (rect[0], rect[1], 0.0)
    box = cv2.boxPoints(rect0)
    pts = np.int0(cv2.transform(np.array([box]), M))[0]
    pts[pts < 0] = 0

    # crop
    img_crop = img_rot[pts[1][1]:pts[0][1],
                       pts[1][0]:pts[2][0]]

    return img_crop

# https://docs.opencv.org/trunk/da/d22/tutorial_py_canny.html
# https://stackoverflow.com/questions/42721213/python-opencv-extrapolating-the-largest-rectangle-off-of-a-set-of-contour-poin
def auto_canny(image, sigma = 0.33):
    # compute the mediam of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    # lower = int(max(0, (1.0 - sigma) * v))
    # upper = int(min(255, (1.0 + sigma) *v))

    # lower = int(0.5 * v)
    # upper = int(max(0, (1.0 - sigma) * v))

    # TODO - Tune paras to defined the boundry for canny algorithm
    lower = int(0.45 * v)
    upper = int(max(0, (1.1 - sigma) * v))

    # # apertureSize = 3
    # L2gradient = True
    edged = cv2.Canny(image, lower, upper)

    # return edged image
    return edged

#https://en.wikipedia.org/wiki/Otsu%27s_method
def otsu_canny(image, thresPath):
    upper, thresh_im = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imwrite(thresPath, thresh_im)
    lower = 0.5 * upper
    edged = cv2.Canny(thresh_im, lower, upper)

    # return edged image
    return edged

def laplacian(image):
    return cv2.Laplacian(image,cv2.CV_64F)

def hough(L, edges):
    # https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/hough_lines/hough_lines.html
    # threshold = 100  # Accumulator threshold parameter. Only those lines are returned that get enough votes ( >threshold ).
    # minLineLength = 0.1*L # The minimum number of points that can form a line. Lines with less than this number of points are disregarded.
    # maxLineGap = 0.02*L # The maximum gap between two points to be considered in the same line.

    # TODO - Tune paras to find hough lines
    threshold = 100  # Accumulator threshold parameter. Only those lines are returned that get enough votes ( >threshold ).
    minLineLength = 0.1*L # The minimum number of points that can form a line. Lines with less than this number of points are disregarded.
    maxLineGap = 0.02*L # The maximum gap between two points to be considered in the same line.

    rho = 1  # The resolution of the parameter r in pixels. We use 1 pixel.
    theta = np.pi / 180  # The resolution of the parameter \theta in radians. We use 1 degree (CV_PI/180)

    lines = cv2.HoughLinesP(image=edges, rho=rho, theta=theta, threshold=threshold,minLineLength = minLineLength, maxLineGap=maxLineGap)
    return lines

def findRectMinArea(src, lines):
    points = []
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(src, (x1, y1), (x2, y2), (0, 0, 255), 3, cv2.LINE_AA)
            points.append((x1, y1))
            points.append((x2, y2))

    array = np.array(points, dtype="float32")

    rect = cv2.minAreaRect(array)
    return rect

def print_line(src, line, color):
    for col1, row1, col2, row2 in line:
        cv2.line(src, (col1, row1), (col2, row2), color, 3, cv2.LINE_AA)

def findLongestLine(src, lines, rowMin, colMin, rowMax, colMax, vertical):
    # TODO - Tune paras to set the threshold for allowed degree
    #threshold = 10
    threshold = 15

    res = None
    resL = 0

    for line in lines:
        angle = 90
        for col1, row1, col2, row2 in line:
            if row1 < rowMin or row1 > rowMax or row2 < rowMin or row2 > rowMax or col1 < colMin or col1 > colMax or col2 < colMin or col2 > colMax:
                continue
            # cv2.line(src, (col1, row1), (col2, row2), (211, 211, 211), 3, cv2.LINE_AA)
            print_line(src, line, (211,211,211))
            if col1 != col2:
                angle = math.atan((row1 - row2) / (col1 - col2))
                angle = abs(angle * 180 / math.pi)
            if vertical:
                if abs(angle - 90) < threshold:
                    L = math.sqrt((row1 - row2)**2 + (col1 - col2)**2)
                    if L > resL:
                        res = line
                        resL = L
            else:
                if abs(angle) < threshold:
                    L = math.sqrt((row1 - row2) ** 2 + (col1 - col2) ** 2)
                    if L > resL:
                        res = line
                        resL = L
    return res


def find_intersection(line1, line2):
    # extract points
    x1, y1, x2, y2 = line1[0].astype(np.longlong)
    x3, y3, x4, y4 = line2[0].astype(np.longlong)
    # compute determinant
    Px = ((x1*y2 - y1*x2)*(x3-x4) - (x1-x2)*(x3*y4 - y3*x4))/  \
        ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
    Py = ((x1*y2 - y1*x2)*(y3-y4) - (y1-y2)*(x3*y4 - y3*x4))/  \
        ((x1-x2)*(y3-y4) - (y1-y2)*(x3-x4))
    # Px = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / ((y4-y3)*(x2-x1) - (x4-x3)*(y2-y1))
    # Py = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / ((y4-y3)*(x2-x1) - (x4-x3)*(y2-y1))
    return Px, Py

def getLineLength(line):
    for col1, row1, col2, row2 in line:
        return  math.sqrt((row1 - row2) ** 2 + (col1 - col2) ** 2)

def isValidPoint(src, point):
    col, row = point
    height = src.shape[0]
    width = src.shape[1]
    if row < 0 or row > height or col <0 or col > width:
        return False
    return True

def findRectDivideConquer(src, lines):
    height = src.shape[0]
    width = src.shape[1]
    # top = findLongestLine(src, lines, 0, 0, height/4, width, vertical=False)
    # bottom = findLongestLine(src, lines, 3*height/4, 0, height, width, vertical=False)
    # left = findLongestLine(src, lines, 0, 0, height, width/4, vertical=True)
    # right = findLongestLine(src, lines, 0, 3*width/4, height, width, vertical=True)

    # TODO - Tune paras to find area
    top = findLongestLine(src, lines, 0, 0, height / 3, width, vertical=False)
    bottom = findLongestLine(src, lines, 2 * height / 3, 0, height, width, vertical=False)
    left = findLongestLine(src, lines, 0, 0, height, width / 3, vertical=True)
    right = findLongestLine(src, lines, 0, 2 * width / 3, height, width, vertical=True)

    if top is None:
        top = np.array((0, 0, width, 0)).reshape(1,4)
    if bottom is None :
        bottom = np.array((0, height, width, height)).reshape(1,4)
    if left is None :
        left = np.array((0, 0, 0, height)).reshape(1,4)
    if right is None:
        right = np.array((width, 0, width, height)).reshape(1,4)

    print_line(src, top, (0, 0, 255))
    print_line(src, bottom, (0, 0, 255))
    print_line(src, left, (0, 0, 255))
    print_line(src, right, (0, 0, 255))

    topLeft = find_intersection(top, left)
    topRight = find_intersection(top, right)
    bottomLeft = find_intersection(bottom, left)
    bottomRight = find_intersection(bottom, right)

    if not isValidPoint(src, topLeft):
        topLeft = 0,0
    if not isValidPoint(src, topRight):
        topRight = width, 0
    if not isValidPoint(src, bottomLeft):
        bottomLeft = 0, height
    if not isValidPoint(src, bottomRight):
        bottomRight = width, height

    return np.array([topLeft, topRight, bottomLeft, bottomRight])


class Extractor:
    
    def __init__(self, params):
        self.points = None
        self.new_points = None
        self.new_seen = 0
        self.params = params
    
    def __call__(self, src, frame):
        return self.extract_whiteboard(src, frame)

    def extract_whiteboard(self, src, frame):
        # Skip if not sampling frame
        if frame % self.params['freq'] != 0:
            if self.points is not None:
                src_cropped = four_point_transform(src, self.points, self.params['width'], self.params['height'])
            else:
                src_cropped = src
            return src_cropped

        # Convert the color from BGR to Gray
        src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

        src_blur = src_gray
        # src_blur = cv2.GaussianBlur(src_gray, (3, 3), 0)

        # Detect edges
        src_edges = auto_canny(src_blur)

        # Detect lines
        l = min(self.params['width'], self.params['height'])
        lines = hough(l, src_edges)
        if lines is None:
            if self.points is not None:
                src_cropped = four_point_transform(src, self.points, self.params['width'], self.params['height'])
            else:
                src_cropped = src
            return src_cropped

        # Find points for cropping
        # rect = findRectMinArea(src, lines)
        points = findRectDivideConquer(src, lines)

        # Update cropping points
        if self.points is not None:
            if np.allclose(self.points, points, atol=self.params['closeness']):
                print("close to _pts")
                self.new_points = None
                self.new_seen = 0
            else:
                if self.new_points is not None:
                    if np.allclose(self.new_points, points, atol=self.params['closeness']):
                        if self.new_seen == 5:
                            print("switch to _newpts")
                            self.points = self.new_points
                            self.new_points = None
                            self.new_seen = 0
                        else:
                            print("close to _newpts")
                            self.new_seen += 1
                    else:
                        print("update _newpts")
                        self.new_points = points
                        self.new_seen = 1
                else:
                    print("init _newpts")
                    self.new_points = points
                    self.new_seen = 1
        else:
            print("init _pts")
            self.points = points
        # self.points = points
        
        # Crop the image
        src_cropped = four_point_transform(src, self.points, self.params['width'], self.params['height'])

        return src_cropped
