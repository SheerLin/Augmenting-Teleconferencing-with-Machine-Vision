import cv2
import numpy as np

from utils import *

class Beautifier:

    def __init__(self, params, debug):
        self.debug = debug

    def __call__(self, src):
        return self.beautify_whiteboard(src)

    def beautify_whiteboard(self, orig):
        src = orig

        # Apply CLAHE for histogram equilization
        src_lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(src_lab)
        clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))
        cl = clahe.apply(l)
        src_clab = cv2.merge((cl,a,b))
        src_clahe = cv2.cvtColor(src_clab, cv2.COLOR_LAB2BGR)

        # Apply smoothening preserving edge
        # src = src_clahe
        # src_filter = cv2.edgePreservingFilter(src, flags=cv2.RECURS_FILTER, sigma_s=60, sigma_r=0.4)

        # Apply sharpening filter
        src = src_clahe
        kernel = np.array([[-1,-1,-1],
                           [-1, 9,-1],
                           [-1,-1,-1]])
        src_sharp = cv2.filter2D(src, -1, kernel)

        src = src_sharp
        return src
