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
    threshold = 100  # Accumulator threshold parameter. Only those lines are returned that get enough votes ( >threshold ).
    minLineLength = 0.1*L # The minimum number of points that can form a line. Lines with less than this number of points are disregarded.
    maxLineGap = 0.02*L # The maximum gap between two points to be considered in the same line.

    rho = 5  # The resolution of the parameter r in pixels. We use 1 pixel.
    theta = np.pi / 180  # The resolution of the parameter \theta in radians. We use 1 degree (CV_PI/180)

    lines = cv2.HoughLinesP(image=edges, rho=rho, theta=theta, threshold=threshold, minLineLength = minLineLength, maxLineGap=maxLineGap)
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
                angle = math.atan((row1 - row2) / (col1 - col2));
                angle = abs(angle * 180 / math.pi);
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
    top = findLongestLine(src, lines, 0, 0, height/3, width, vertical=False)
    bottom = findLongestLine(src, lines, 2*height/3, 0, height, width, vertical=False)
    left = findLongestLine(src, lines, 0, 0, height, width/3, vertical=True)
    right = findLongestLine(src, lines, 0, 2*width/3, height, width, vertical=True)

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
        self.area = self.params['width'] * self.params['height']

    def __call__(self, src, frame):
        return self.extract_whiteboard(src, frame)

    def pipelineA(self, src):
        orig = src.copy()

        # Lower the contrast
        # alpha 1  beta 0      --> no change
        # 0 < alpha < 1        --> lower contrast
        # alpha > 1            --> higher contrast
        # -127 < beta < +127   --> good range for brightness values
        alpha = 0.7
        beta = -10
        src_contrast = cv2.addWeighted(src, alpha, src, 0, beta)
        # cv2.imshow("Contrast", src_contrast)

        # Apply Color Filter
        src = src_contrast
        b = []
        g = []
        r = []
        for h in range(100, 140): # 480
            for w in range(300, 340): # 640
                b.append(src[h][w][0])
                g.append(src[h][w][1])
                r.append(src[h][w][2])
                for k in range(3):
                    src[h][w][k] = 255
        avg_b = sum(b)/len(b)
        avg_g = sum(g)/len(g)
        avg_r = sum(r)/len(r)
        delta = 35
        m = 1.5
        lower = np.array([avg_b-delta, avg_g-delta, avg_r-delta], dtype = "uint8")
        upper = np.array([avg_b+m*delta, avg_g+m*delta, avg_r+m*delta], dtype = "uint8")
        mask = cv2.inRange(src, lower, upper)
        src_white = cv2.bitwise_and(src, src, mask = mask)
        cv2.imshow('White', src_white)

        # Apply Morphing
        src = src_white
        morph = src
        kernel = np.ones((5, 5), np.uint8)
        iterations = 2
        morph = cv2.dilate(morph, kernel, iterations=iterations)
        morph = cv2.erode(morph, kernel, iterations=iterations)
        cv2.imshow('Open', morph)

        morph = cv2.erode(morph, kernel, iterations=iterations)
        morph = cv2.dilate(morph, kernel, iterations=iterations)
        cv2.imshow('Close', morph)

        # Convert the color from BGR to Gray
        src = morph
        src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Gray", src_gray)

        # Apply blurring to smoothen
        src = src_gray
        src_blur = cv2.GaussianBlur(src, (5, 5), 3, borderType=cv2.BORDER_ISOLATED)
        cv2.imshow("Blur", src_blur)

        # Apply mean adaptive theshold
        src = src_blur
        src_thresh = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 3)
        cv2.imshow("Thresh", src_thresh)

        _, src_thresh2 = cv2.threshold(src, 70, 255, cv2.THRESH_BINARY)
        cv2.imshow("Thresh2", src_thresh2)

        # Apply Erosion
        src = src_thresh
        kernel = np.ones((5, 5), np.uint8)
        iterations = 1
        img_erosion = cv2.erode(src, kernel, iterations=iterations+1)
        cv2.imshow('Erosion2', img_erosion)
        # Apply Dilation
        src = img_erosion
        img_dilation = cv2.dilate(src, kernel, iterations=iterations)
        cv2.imshow('Dilation2', img_dilation)

        # Apply Edgedetection
        # edges = cv2.Canny(src, 50, 70)
        # cv2.imshow('Edges', edges)

        # Find contours
        src = img_dilation
        contours, _ = cv2.findContours(src, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key = cv2.contourArea)
        # Find approx rectangle contours
        for c in contours[-5:]:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.04 * peri, True)
            edges = len(approx)
            # if edges >= 3 and edges <= 5:
            cv2.drawContours(orig, [c], 0, (0, 255, 0), 2)
        cv2.imshow("Contours", orig)

    def threshold(self, src):
        orig = src

        ##################
        # begin threshold
        ##################

        # Threshold the HSV image to get only white colors
        src = orig
        lower_white = np.array([0,0,0], dtype=np.uint8)
        upper_white = np.array([150,100,255], dtype=np.uint8)
        hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, lower_white, upper_white)
        thresh_hsv = cv2.bitwise_and(src, src, mask=mask)
        cv2.imshow("hsv", hsv)
        # cv2.imshow('Thresh HSV', np.hstack([hsv, thresh_hsv]))



        src = orig
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        kernel = np.ones((5, 5))
        iterations = 2
        # morph = cv2.morphologyEx(src, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        # morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel, iterations=iterations)
        morph = src
        morph = cv2.erode(morph, kernel, iterations=iterations)
        morph = cv2.dilate(morph, kernel, iterations=iterations)
        cv2.imshow('morph', morph)

        src = morph
        _, thresh = cv2.threshold(src, 128, 0, cv2.THRESH_TOZERO)
        cv2.imshow('Thresh', thresh)

        src = thresh
        hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
        cv2.imshow("hsv", hsv)

        src = morph
        channels = np.split(np.asarray(src), 3, axis=2)
        height, width, _ = channels[0].shape
        for i in range(3):
            _, channel = cv2.threshold(channels[0], 128, 0, cv2.THRESH_TOZERO)
            channels[i] = np.reshape(channel, newshape=(height, width, 1))
        # merge the channels
        channels = np.concatenate(channels, axis=2)
        cv2.imshow("channels", channels)


        # Remove reflection
        src = orig
        # _, src_thresh = cv2.threshold(src, 200, 0, cv2.THRESH_TRUNC)
        _, mask = cv2.threshold(src, 200, 128, cv2.THRESH_BINARY)
        _, src_thresh = cv2.threshold(src, 50, 0, cv2.THRESH_TRUNC)
        # src_thresh = cv2.bitwise_and(src, mask)
        # cv2.imshow("Reflection2", mask)
        # cv2.imshow("Reflection", src_thresh)

    def morph(self, src):
        #############
        # begin morph
        #############

        morph = src.copy()
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel, iterations=3)
        cv2.imshow("close", morph)
        morph = cv2.morphologyEx(morph, cv2.MORPH_OPEN, kernel)
        cv2.imshow("open", morph)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

        # take morphological gradient
        gradient_image = cv2.morphologyEx(morph, cv2.MORPH_GRADIENT, kernel)
        cv2.imshow("grad", gradient_image)

        # split the gradient image into channels
        image_channels = np.split(np.asarray(gradient_image), 3, axis=2)
        channel_height, channel_width, _ = image_channels[0].shape
        # apply Otsu threshold to each channel
        for i in range(0, 3):
            _, image_channels[i] = cv2.threshold(~image_channels[i], 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
            image_channels[i] = np.reshape(image_channels[i], newshape=(channel_height, channel_width, 1))
        # merge the channels
        image_channels = np.concatenate((image_channels[0], image_channels[1], image_channels[2]), axis=2)

        # save the denoised image
        cv2.imshow("morph", image_channels)

    def hough(self, src):
        ###############
        # canny + hough
        ###############

        # Apply Canny
        src = src_thresh
        src_edges = auto_canny(src)
        cv2.imshow("Edges", src_edges)

        # Apply Hough
        src = src_thresh
        lines = hough(l, src_thresh)
        findRectDivideConquer(src, lines)
        cv2.imshow("Lines", src)

        src = src_thresh
        lines = hough(l, src_edges)
        findRectDivideConquer(src, lines)
        cv2.imshow("Lines2", src)

        # Detect edges
        src_edges = auto_canny(src_contrast)
        cv2.imshow("Edges", src_edges)

        # Detect lines
        lines = hough(l, src_edges)
        if lines is None:
            if self.points is not None:
                src_cropped = four_point_transform(src, self.points, self.params['width'], self.params['height'])
            else:
                src_cropped = src
            return src_cropped

        # Find points for cropping
        rect = findRectMinArea(src, lines)
        points = findRectDivideConquer(src, lines)
        cv2.imshow("Hough", src)

    def contourcrop(self, src):
        ##############
        # contour crop
        ##############

        # contours = filter(self.filterLargeContour, contours)

        # Create mask where white is what we want, black otherwise
        # mask = np.zeros_like(src_thresh_inv1)
        # Draw filled contour in mask
        # cv2.drawContours(mask, [c], 0, 255, -1)
        # cv2.imshow("Mask", mask)

        # x, y, h, w = cv2.boundingRect(c)
        # lines1 = hough(l, mask)
        # findRectDivideConquer(src1, lines1)
        # cv2.imshow("Lines1", src1)

        # # Extract out the object and place into output image
        # out = np.zeros_like(src_thresh_inv1)
        # out[mask == 255] = src_thresh_inv1[mask == 255]

        # # Now crop
        # print(mask)
        # (y, x) = np.where(mask == 255)
        # (topy, topx) = (np.min(y), np.min(x))
        # (bottomy, bottomx) = (np.max(y), np.max(x))
        # out = out[topy:bottomy+1, topx:bottomx+1]

        # # Show the output image
        # cv2.imshow('Output', out)

        # c2 = c[:,0,:]
        # x, y, h, w = cv2.boundingRect(c)
        # c3 = np.array([[y+h, x], [y+h, x+w], [y, x+w], [y, x]])
        # cv2.rectangle(src1, (y+h, x), (y, x+w), (255,0,0), 2)
        # cv2.imshow("Contours", src1)
        # # print("cr3", c3)
        # src_cropped1 = four_point_transform(src1, c3, self.params['width'], self.params['height'])
        # cv2.imshow("Cropped", src_cropped1)
        # # raw_input()
        pass

    def updatepoints(self):
        # # Update cropping points
        # if self.points is not None:
        #     if np.allclose(self.points, points, atol=self.params['closeness']):
        #         print("close to pts")
        #         self.new_points = None
        #         self.new_seen = 0
        #     else:
        #         if self.new_points is not None:
        #             if np.allclose(self.new_points, points, atol=self.params['closeness']):
        #                 if self.new_seen == 5:
        #                     print("### switch to newpts ###")
        #                     self.points = self.new_points
        #                     self.new_points = None
        #                     self.new_seen = 0
        #                 else:
        #                     print("close to newpts")
        #                     self.new_seen += 1
        #             else:
        #                 print("update newpts")
        #                 self.new_points = points
        #                 self.new_seen = 1
        #         else:
        #             print("init newpts")
        #             self.new_points = points
        #             self.new_seen = 1
        # else:
        #     print("init _pts")
        #     self.points = points
        # self.points = points

        # print("points", points)
        pass

    def pipelineB(self, src):
        orig = src.copy()

        # Apply blurring to sharpen
        src = orig.copy()
        src_blur = cv2.GaussianBlur(src, (5, 5), 3, borderType=cv2.BORDER_ISOLATED)
        src_sharp1 = cv2.addWeighted(src, 1.5, src_blur, -0.7, 0)
        cv2.imshow("Sharp1", src_sharp1)

        # Apply sharpening filter
        src = orig.copy()
        kernel = np.array([[-1,-1,-1],
                           [-1, 9,-1],
                           [-1,-1,-1]])
        src_sharp = cv2.filter2D(src, -1, kernel)
        cv2.imshow("Sharp", src_sharp)


        # Apply smoothening preserving edge
        # src = src_sharp
        # src_filter = cv2.edgePreservingFilter(src, flags=cv2.RECURS_FILTER, sigma_s=60, sigma_r=0.4)
        # cv2.imshow("Filter", src_filter)

        src = src_sharp
        b = []
        g = []
        r = []
        for h in range(100, 140): # 480
            for w in range(300, 340): # 640
                b.append(src[h][w][0])
                g.append(src[h][w][1])
                r.append(src[h][w][2])
                for k in range(3):
                    src[h][w][k] = 255
        avg_b = sum(b)/len(b)
        avg_g = sum(g)/len(g)
        avg_r = sum(r)/len(r)
        delta = 60
        m = 1.5
        lower = np.array([max(0, avg_b-delta), max(0, avg_g-delta), max(0, avg_r-delta)], dtype = "uint8")
        upper = np.array([min(255, avg_b+m*delta), min(255, avg_b+m*delta), min(255, avg_b+m*delta)], dtype = "uint8")
        mask = cv2.inRange(src, lower, upper)
        src_white = cv2.bitwise_and(src, src, mask = mask)
        cv2.imshow('White', src_white)

        # Convert the color from BGR to Gray
        src = src_white
        src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Gray", src_gray)

        src = src_gray
        morph = src
        kernel = np.ones((3, 3), np.uint8)
        iterations = 2
        morph = cv2.erode(morph, kernel, iterations=iterations+1)
        morph = cv2.dilate(morph, kernel, iterations=iterations)
        cv2.imshow('Morph Open', morph)

        # Apply smoothening preserving edge
        src = morph
        src_filter = cv2.edgePreservingFilter(src, flags=cv2.RECURS_FILTER, sigma_s=60, sigma_r=0.4)
        cv2.imshow("Filter", src_filter)

        # Apply mean adaptive theshold
        src = src_filter
        src_thresh = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 7)
        cv2.imshow("Thresh", src_thresh)

        src = src_thresh
        morph = src
        kernel = np.ones((3, 3), np.uint8)
        iterations = 1
        morph = cv2.erode(morph, kernel, iterations=iterations+1)
        morph = cv2.dilate(morph, kernel, iterations=iterations)
        cv2.imshow('Morphe Open 2', morph)

        # Find contours
        src = morph
        contours, _ = cv2.findContours(src, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        contours = sorted(contours, key = cv2.contourArea)
        # Find approx rectangle contours
        for c in contours[-3:]:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.03 * peri, True)
            area = cv2.contourArea(c)
            edges = len(approx)
            if edges >= 3 and edges <= 5 and area > 0.03 * 640 * 480 and area < 0.9 * 640 * 480:
                cv2.drawContours(orig, [c], 0, (0, 255, 0), 2)
                x,y,w,h = cv2.boundingRect(c)
                cv2.rectangle(orig, (x,y), (x+w,y+h), (255,0,0), 2)
            # else:
            #     cv2.drawContours(orig, [c], 0, (255, 0, 0), 2)
        cv2.imshow("Contours", orig)
        # raw_input()

    def extract_whiteboard(self, src, frame):
        # # Skip if not sampling frame
        # if frame % self.params['freq'] != 0:
        #     if self.points is not None:
        #         src_cropped = four_point_transform(src, self.points, self.params['width'], self.params['height'])
        #     else:
        #         src_cropped = src
        #     return src_cropped

        l = min(self.params['width'], self.params['height'])
        orig = src
        cv2.imshow('Video', orig)

        src = orig.copy()
        self.pipelineB(src)

        # Crop the image
        # src_cropped = four_point_transform(src, self.points, self.params['width'], self.params['height'])
        return src
