import time

import cv2
import numpy as np

import beautifier
import extractor
import undistortion

SAMPLE_FREQ = 10
MEAN_LENGTH  = 10
BENCHMARK = True

class Engine:

    def __init__(self, width, height, cam_device_number):
        self.width = width
        self.height = height
        self.cam_device_number = cam_device_number
        self.frame_num = 0
        self.start_time = 0
        self.end_time = 0
        
        self.extractor = extractor.Extractor({
            'width': self.width,
            'height': self.height,
            'freq': SAMPLE_FREQ,
            'closeness': 20,
            'center': (300, 100, 40, 40),
            'benchmark': BENCHMARK
        })
        self.undistorter = undistortion.Undistortion(
            cam_device_number=self.cam_device_number
        )
        self.beautifier = beautifier.Beautifier({

        })

        self.run_pre = []
        self.run_pre_sum = np.zeros((height, width, 3))

        self.run_post = []
        self.run_post_sum = np.zeros((height, width, 3))

    def average(self, src, run, run_sum):
        if len(run) == MEAN_LENGTH:
            # Discard first frame, and subtract
            last = run.pop(0)
            run_sum = np.subtract(run_sum, last)
        
        # Save new frame, and add
        run.append(src)
        run_sum = np.add(run_sum, src)
        
        # Compute average
        src = np.divide(run_sum, len(run))
        src = src.astype(np.uint8)
        
        # src = np.average(np.array(run), axis=0).astype(np.uint8)
        return src, run_sum

        # vanilla           28.2 fps
        # avg1              24.5 fps
        # avg2              22   fps
        # undistorter        5.5 fps
        # extractor          6.5 fps   1 freq
        # extractor         22   fps  10 freq
        # beautifier         9   fps
        # whole            3.3   fps

        # no u               6.8 fps
        # no b               3.8 fps
        # no u/b avg2       17   fps
        # no u/b avg1       19.5 fps
    def time(self):
        start_time = self.start_time
        end_time = time.time()
        c = 25
        if self.frame_num % c == 0:
            if start_time != 0:
                t = end_time - start_time
                fps = c / t
                print("fps: {:.2f}".format(fps))
            self.start_time = end_time
    
    def process(self, orig):
        src = orig.copy(order='C')
        # cv2.imshow('orig', src)
        
        src, self.run_pre_sum = self.average(src, self.run_pre, self.run_pre_sum)

        src = self.undistorter(src).copy(order='C')        
        # cv2.imshow('undistorter', src)
        
        src = self.extractor(src, self.frame_num)
        # cv2.imshow('extractor', src)

        src = self.beautifier(src)

        # src, self.run_post_sum = self.average(src, self.run_post, self.run_post_sum)
        
        self.time()
        self.frame_num += 1

        # show = np.hstack([orig, src])
        # cv2.imshow('Video', show)

        cv2.imshow('Orig', orig)
        # cv2.imshow('Out', src)
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
