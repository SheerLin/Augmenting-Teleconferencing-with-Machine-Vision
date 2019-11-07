import cv2
import numpy as np

from utils import *

class Extractor:

    def __init__(self, params):
        self.params = params
        self.points = None
        self.dims = None
        self.new_points = None
        self.new_dims = None
        self.new_seen = 0
        self.dims_center = params['center']
        self.width = params['width']
        self.height = params['height']
        self.area = self.width * self.height
        self.logfile = None
        if params['benchmark']:

            self.logfile = open("extract_points.log", "w")

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
        # cv2.imshow('Sharp', src_sharp)

        # TODO check HUE space
        # Filter out other colors
        src = src_sharp
        delta = 60
        m = 1.5
        pixels = src[center_y:center_y+center_box_h, center_x:center_x+center_box_w].copy()
        pixels = pixels.reshape((pixels.shape[1] * pixels.shape[1], pixels.shape[2]))
        avg = pixels.mean(axis=0).reshape((3))
        lower = np.maximum(avg-delta, 0).astype('uint8')
        upper = np.minimum(avg+delta*m, 255).astype('uint8')
        mask = cv2.inRange(src, lower, upper)
        src_white = cv2.bitwise_and(src, src, mask = mask)
        # cv2.imshow('White', src_white)

        # Convert the color from BGR to Gray
        src = src_white
        src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        # cv2.imshow('Gray', src_gray)

        # Erosion followed by Dilation (Opening Morph)
        # Reduces noise
        morph = src_gray
        kernel = np.ones((3, 3), np.uint8)
        iterations = 3
        morph = cv2.erode(morph, kernel, iterations=iterations)
        morph = cv2.dilate(morph, kernel, iterations=iterations)
        # cv2.imshow('Morph Open', morph)

        # Apply smoothening preserving edge
        src = morph
        src_filter = cv2.edgePreservingFilter(src, flags=cv2.RECURS_FILTER, sigma_s=60, sigma_r=0.4)
        # cv2.imshow('Filter', src_filter)

        # Apply mean adaptive theshold
        # Convert to edges
        src = src_filter
        src_thresh = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 7)
        # cv2.imshow('Thresh', src_thresh)

        # Erosion followed by Dilation (Opening Morph)
        # Connect breaks caused by noise
        morph = src_thresh
        kernel = np.ones((3, 3), np.uint8)
        iterations = 1
        morph = cv2.erode(morph, kernel, iterations=iterations+1)
        morph = cv2.dilate(morph, kernel, iterations=iterations)
        # cv2.imshow('Morphe Open 2', morph)

        # Find and draw big, rectangular contours (connected regions)
        src = morph
        src_ex = orig
        contours, _ = cv2.findContours(src, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        src_ex, (x, y, w, h) = self.find_wb_contour(src_ex, contours)

        # Draw color filter box
        # debug
        cv2.rectangle(src_ex, (center_x,center_y), (center_x+center_box_w,center_y+center_box_h), (0,0,255), 2)
        
        return src_ex, (x,y,w,h)

    def detect_wb_edges(self, orig, dims, width, height):
        (x,y,w,h) = dims

        # Crop
        src = orig
        src_crop = src[y:y+h, x:x+w]
        
        src = src_crop
        src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)	
        # cv2.imshow("Gray", src_gray)        
        
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
        # cv2.imshow("Edges", src_edges)

        # Use Hough to detect lines
        # https://docs.opencv.org/2.4/doc/tutorials/imgproc/imgtrans/hough_lines/hough_lines.html
        src = src_edges
        l = min(width, height)
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
        # cv2.imshow("Hough", src_hough)

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
                # debug
                cv2.drawContours(src_ex, [c], 0, (0, 255, 0), 2)
                cv2.rectangle(src_ex, (x,y), (x+w,y+h), (255,0,0), 2)
                break
        return src_ex, (x,y,w,h)

    def update_points(self, points, dims, closeness):
        if self.points is not None:
            saved_points = self.points + self.dims[:2]
            cur_points = points + dims[:2]
            
            if np.allclose(saved_points, cur_points, atol=closeness):
                print('### Close Points ###')
                self.new_points = None
                self.new_dims = None
                self.new_seen = 0
            else:
                if self.new_points is not None:
                    new_points = self.new_points + self.new_dims[:2]
                    if np.allclose(new_points, cur_points, atol=closeness):
                        if self.new_seen == 3:
                            print('>>> Switch Points <<<')
                            self.points = points
                            self.dims = dims
                            self.new_points = None
                            self.new_dims = None
                            self.new_seen = 0
                        else:
                            # print('close to newpts')
                            self.new_seen += 1
                    else:
                        # print('update newpts')
                        self.new_points = points
                        self.new_dims = dims
                        self.new_seen = 1
                else:
                    # print('init newpts')
                    self.new_points = points
                    self.new_dims = dims
                    self.new_seen = 1
        else:
            # print('init _pts')
            self.points = points
            self.dims = dims

    def crop(self, orig):
        (x,y,w,h) = self.dims
        src = orig.copy()
        src = src[y:y+h, x:x+w]
        src = four_point_transform(src, self.points, self.width, self.height)

        if self.logfile:
            for i in range(len(self.points)):
                self.logfile.write(str(self.points[i][0]) + "," + str(self.points[i][1]) + ";")
            self.logfile.write("\n")
        
        return src

    def extract_whiteboard(self, orig, frame_num):
        src = orig.copy()
        width = self.params['width']
        height = self.params['height']
        
        # Skip processing if not sampling frame
        if frame_num % self.params['freq'] != 0:
            if self.points is not None:
                src_ex = self.crop(orig)
            else:
                src_ex = orig
            return src_ex

        # Detect whiteboard
        src_a, dims = self.detect_wb_contour(src)
        src_b, points = self.detect_wb_edges(src, dims, width, height)
        
        if points is not None:
            self.update_points(points, dims, self.params['closeness'])
            
        # Extract whiteboard
        if self.points is not None:
            src_ex = self.crop(orig)
            show = np.hstack([src_a, src_b])
            # cv2.imshow('Processing', show)
            return src_ex

        # Failed detection/extraction
        # Return original
        return src
