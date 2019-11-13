import cv2
import numpy as np

from utils import *

class Beautifier:

    def __init__(self, params):
        pass

    def __call__(self, src):
        return self.beautify_whiteboard(src)

    def beautify_whiteboard(self, orig):
        # src = orig.copy()
        src = orig

        # Apply CLAHE for histogram equilization
        src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(src_lab)
        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))
        cl = clahe.apply(l)
        src_clab = cv2.merge((cl,a,b))
        src_clahe = cv2.cvtColor(src_clab, cv2.COLOR_LAB2BGR)
        # cv2.imshow('Clahe', src_clahe)

        # Apply smoothening preserving edge
        # src = src_clahe
        # src_filter = cv2.edgePreservingFilter(src, flags=cv2.RECURS_FILTER, sigma_s=60, sigma_r=0.4)
        # cv2.imshow('Filter', src_filter)

        # TODO: Sharp?
        # Apply sharpening filter
        src = src_clahe
        kernel = np.array([[-1,-1,-1],
                           [-1, 9,-1],
                           [-1,-1,-1]])
        src_sharp = cv2.filter2D(src, -1, kernel)
        # cv2.imshow('Sharp', src_sharp)

        # TODO: Gray or Colored?
        # Convert the color from BGR to Gray
        # src = src_clahe
        # src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
  
        # TODO: Threshold?
        # Apply mean adaptive theshold
        # src = src_gray
        # src_thresh = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #         cv2.THRESH_BINARY, 11, 3)
        # cv2.imshow('Thresh', src_thresh)

        # Erosion followed by Dilation (Opening Morph)
        # Reduces noise
        # morph = src_clahe
        # kernel = np.ones((3, 3), np.uint8)
        # iterations = 1
        # morph = cv2.erode(morph, kernel, iterations=iterations)
        # # cv2.imshow('Morph Open', morph)
        # morph = cv2.dilate(morph, kernel, iterations=iterations)

        
        src = src_sharp
        return src
