import cv2
import numpy as np

from utils import *

class Beautifier:

    def __init__(self, params):
        pass

    def __call__(self, src):
        return self.beautify_whiteboard(src)

    def beautify_whiteboard(self, orig):
        src = orig.copy()

        # Apply CLAHE for histogram equilization
        src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(src_lab)
        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))
        cl = clahe.apply(l)
        src_clab = cv2.merge((cl,a,b))
        src_clahe = cv2.cvtColor(src_clab, cv2.COLOR_LAB2BGR)
        # cv2.imshow('Clahe', src_clahe)

        # Convert the color from BGR to Gray
        src = src_clahe
        src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

        # Apply mean adaptive theshold
        src = src_gray
        src_thresh = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY, 11, 3)
        # cv2.imshow('Thresh', src_thresh)
        
        src = src_thresh
        return src
