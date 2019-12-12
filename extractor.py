import logging

import cv2
import numpy as np

from utils import *

class Extractor:

    def __init__(self, params, benchmark, debug):
        self.logger = logging.getLogger("ATCV")
        
        self.width = params['width']
        self.height = params['height']
        self.freq = params['freq']
        self.dims_center = params['center']
        self.closeness = params['closeness']
        self.area = self.width * self.height
        
        self.points = None
        self.dims = None
        self.new_points = None
        self.new_dims = None
        self.new_seen = 0
        
        self.benchmark = benchmark
        if benchmark:
            self.logfile = open("extract_points.log", "w")
        
        self.debug = debug
        self.show_image = debug

    def __call__(self, src, frame_num):
        return self.extract_whiteboard(src, frame_num)

    def detect_wb_contour(self, orig):
        (center_x, center_y, center_box_w, center_box_h) = self.dims_center
    
        # Apply sharpening filter
        src = orig
        kernel = np.array([[-1,-1,-1],
                           [-1, 9,-1],
                           [-1,-1,-1]])
        src_sharp = cv2.filter2D(src, -1, kernel)

        # Apply CLAHE for histogram equilization
        src = src_sharp
        src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(src_lab)
        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        src_clab = cv2.merge((cl, a, b))
        src_clahe = cv2.cvtColor(src_clab, cv2.COLOR_LAB2BGR)

        # TODO check HUE space
        # Filter out other colors
        src = src_clahe
        delta = 60
        m = 1.5
        pixels = src[center_y:center_y+center_box_h, center_x:center_x+center_box_w].copy()
        pixels = pixels.reshape((pixels.shape[0] * pixels.shape[1], pixels.shape[2]))
        avg = pixels.mean(axis=0).reshape((3))
        lower = np.maximum(avg-delta, 0).astype('uint8')
        upper = np.minimum(avg+delta*m, 255).astype('uint8')
        mask = cv2.inRange(src, lower, upper)
        src_white = cv2.bitwise_and(src, src, mask = mask)

        # Convert the color from BGR to Gray
        src = src_white
        src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

        # Erosion followed by Dilation (Opening Morph)
        # Reduces noise
        morph = src_gray
        kernel = np.ones((3, 3), np.uint8)
        iterations = 3
        morph = cv2.erode(morph, kernel, iterations=iterations)
        morph = cv2.dilate(morph, kernel, iterations=iterations)

        # Apply smoothening preserving edge
        src = morph
        src_filter = cv2.edgePreservingFilter(src, flags=cv2.RECURS_FILTER, sigma_s=60, sigma_r=0.4)

        # Apply mean adaptive theshold
        # Convert to edges
        src = src_filter
        src_thresh = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 7)

        # Erosion followed by Dilation (Opening Morph)
        # Connect breaks caused by noise
        morph = src_thresh
        kernel = np.ones((3, 3), np.uint8)
        iterations = 1
        morph = cv2.erode(morph, kernel, iterations=iterations+1)
        morph = cv2.dilate(morph, kernel, iterations=iterations)

        # Find and draw big, rectangular contours (connected regions)
        src = morph
        src_ex = orig
        contours, _ = cv2.findContours(src, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        src_ex, (x, y, w, h) = self.find_wb_contour(src_ex, contours)

        if self.debug:
            # Draw color filter box
            cv2.rectangle(src_ex, (center_x,center_y), (center_x+center_box_w,center_y+center_box_h), (0,0,255), 2)
        
        return src_ex, (x,y,w,h)

    def detect_wb_edges(self, orig, dims):
        (x,y,w,h) = dims

        # Crop
        src = orig
        src_crop = src[y:y+h, x:x+w]
        
        src = src_crop
        src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)	
        
        src = src_gray
        src_blur = cv2.GaussianBlur(src, (3, 3), 0)

        # Use Canny to detect edges
        # https://docs.opencv.org/trunk/da/d22/tutorial_py_canny.html
        # https://stackoverflow.com/questions/42721213/python-opencv-extrapolating-the-largest-rectangle-off-of-a-set-of-contour-poin
        src = src_blur
        v = np.median(src)
        sigma = 0.33
        lower = int(0.45 * v)
        upper = int(max(0, (1.1 - sigma) * v))
        src_edges = cv2.Canny(src, lower, upper)

        # Use Hough to detect lines
        # https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/hough_lines/hough_lines.html
        src = src_edges
        l = min(self.width, self.height)
        threshold = 100  # Accumulator threshold parameter. Only those lines are returned that get enough votes ( >threshold ).
        minLineLength = 0.1*l # The minimum number of points that can form a line. Lines with less than this number of points are disregarded.
        maxLineGap = 0.02*l # The maximum gap between two points to be considered in the same line.
        rho = 5  # The resolution of the parameter r in pixels. We use 1 pixel.
        theta = np.pi / 180  # The resolution of the parameter \theta in radians. We use 1 degree (CV_PI/180)
        lines = cv2.HoughLinesP(src_edges, rho, theta, threshold, None, minLineLength, maxLineGap)
        if lines is None:
            points = np.array([[x, y], [x+w, y], [x, y+h], [x+w, y+h]])
            return orig, points

        # Find points for cropping
        src_hough = src_crop
        points = find_rect(src_hough, lines)

        src_ex = orig
        src_ex[y:y+h, x:x+w] = src_hough[:,:]

        return src_ex, points

    def find_wb_contour(self, src_ex, contours):
        x,y,w,h = 0, 0, self.width, self.height
        center_x, center_y, center_box_w, center_box_h = self.dims_center
        contours = sorted(contours, key = cv2.contourArea)

        for c in contours[-5:]:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.03 * peri, True)
            area = cv2.contourArea(c)
            edges = len(approx)
            x,y,w,h = cv2.boundingRect(c)
            
            is_rectangular = edges >= 3 and edges <= 5
            is_big = area > 0.15 * self.area and area < 0.8 * self.area
            is_center_x = x <= center_x+center_box_w/2 and x+w >= center_x+center_box_w/2
            is_center_y = y <= center_y + center_box_h/2 and y+h >= center_y+center_box_h/2

            if is_rectangular and is_big and is_center_x and is_center_y:
                if self.debug:
                    # Draw detected contour and bounding box
                    cv2.drawContours(src_ex, [c], 0, (0, 255, 0), 2)
                    cv2.rectangle(src_ex, (x,y), (x+w,y+h), (255,0,0), 2)
                break
        return src_ex, (x,y,w,h)

    def update_points(self, points, dims):
        threshold = 1
        if self.points is not None:
            saved_points = self.points + self.dims[:2]
            cur_points = points + dims[:2]
            
            if np.allclose(saved_points, cur_points, atol=self.closeness):
                self.logger.debug('### Close Points ###')
                self.new_points = None
                self.new_dims = None
                self.new_seen = 0
            else:
                if self.new_points is not None:
                    new_points = self.new_points + self.new_dims[:2]
                    if np.allclose(new_points, cur_points, atol=self.closeness):
                        if self.new_seen == threshold:
                            self.logger.debug('>>> Switch Points <<<')
                            self.points = points
                            self.dims = dims
                            self.new_points = None
                            self.new_dims = None
                            self.new_seen = 0
                        else:
                            self.new_seen += 1
                    else:
                        self.new_points = points
                        self.new_dims = dims
                        self.new_seen = 1
                else:
                    self.new_points = points
                    self.new_dims = dims
                    self.new_seen = 1
        else:
            self.points = points
            self.dims = dims

    def crop(self, orig):
        (x,y,w,h) = self.dims
        src = orig.copy()
        src = src[y:y+h, x:x+w]
        src = four_point_transform(src, self.points, self.width, self.height)

        if self.benchmark:
            for i in range(len(self.points)):
                self.logfile.write(str(self.points[i][0]+x) + "," + str(self.points[i][1]+y) + ";")
            self.logfile.write("\n")
        
        # np_points = np.array(self.points, np.int)[[0,1,3,2],:]
        # np_points[:,0] = np_points[:,0] + self.dims[0]
        # np_points[:,1] = np_points[:,1] + self.dims[1]

        # detected = orig.copy()
        # cv2.polylines(detected, [np_points], True, (0,255,0), thickness=3)
        # detected = cv2.resize(detected, (0, 0), fx=0.5, fy=0.5)
        # cv2.imshow('detected wb', detected)
        return src

    def extract_whiteboard(self, orig, frame_num):
        src = orig.copy()
        
        # Skip processing if not sampling frame
        if frame_num % self.freq != 0:
            if self.points is not None:
                src_ex = self.crop(orig)
            else:
                src_ex = orig
            return src_ex

        # Detect whiteboard
        src_a, dims = self.detect_wb_contour(src)
        src_b, points = self.detect_wb_edges(src, dims)
        if self.show_image:
            cv2.imshow('Processing', src_b)
        
        if points is not None:
            self.update_points(points, dims)
        
        # Extract whiteboard
        if self.points is not None:
            src_ex = self.crop(orig)
            return src_ex

        # Failed detection/extraction
        # Return original
        return src
