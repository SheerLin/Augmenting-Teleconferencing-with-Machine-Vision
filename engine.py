import cv2
import numpy as np

import beautifier
import extractor
import undistortion

SAMPLE_FREQ = 10
MEAN_LENGTH  = 10

class Engine:

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.frame_num = 0
        self.run_pre = []
        self.run_post = []

        self.extractor = extractor.Extractor({
            'width': self.width,
            'height': self.height,
            'freq': SAMPLE_FREQ,
            'closeness': 20
        })
        # self.undistort_instance = undistortion.Undistortion()
        self.beautifier = beautifier.Beautifier({

        })

    def average(self, src, run):
        if len(run) == MEAN_LENGTH:
            # Discard first frame
            run.pop(0)
        # Add new frame
        run.append(src)
        
        # Compute average frame
        # weights=range(10)
        src = np.average(np.array(run), axis=0).astype(np.uint8)
        
        return src
    
    def process(self, orig):
        src = orig.copy()
        cv2.imshow('oirg', src)
        
        src = self.average(src, self.run_pre)
        
        # src = self.undistort_instance(src)
        # cv2.imshow('undistorter', src)

        src = self.extractor(src, self.frame_num)
        # cv2.imshow('extractor', src)

        src = self.beautifier(src)
        # cv2.imshow('beautifier', src)

        # src = self.average(src, self.run_post)
        
        self.frame_num += 1
        show = np.hstack([orig, src])
        cv2.imshow('Video', show)
        return src

####################
# begin denoising
####################


# b,g,r = cv2.split(src)           # get b,g,r
# rgb_img = cv2.merge([r,g,b])     # switch it to rgb

# Denoising
# dst = cv2.fastNlMeansDenoisingColored(src,None,10,10,7,21)
# dst = cv2.medianBlur(src, 3)

# b,g,r = cv2.split(dst)           # get b,g,r
# rgb_dst = cv2.merge([r,g,b])     # switch it to rgb
# cv2.imshow("a", dst)

# fastNlMeansDenoisingColoredMulti()
