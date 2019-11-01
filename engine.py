import cv2
import numpy as np

import beautifier
import extractor
import undistortion

class Engine:

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.frame_num = 0
        self.extractor = extractor.Extractor({
            'width': self.width,
            'height': self.height,
            'freq': 10,
            'closeness': 20
        })
        # self.undistort_instance = undistortion.Undistortion(profile_path=None,
        #                                                     chessboard_folder_path=undistortion.default_chessboard_path)
        self.beautifier = beautifier.Beautifier({

        })

    def process(self, orig):
        src = orig.copy()
        
        # src = self.undistort_instance(src)

        src = self.extractor(src, self.frame_num)
        # cv2.imshow('extractor', src)

        src = self.beautifier(src)
        # cv2.imshow('beautifier', src)

        self.frame_num += 1
        #show = np.hstack([orig, src])
        #cv2.imshow('Video', show)
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
