import logging
import time

import cv2
import numpy as np
import math

import beautifier
import extractor
import undistortion

SAMPLE_FREQ = 10
MEAN_LENGTH  = 10
BENCHMARK = True
SHOW_IMAGE = True

class Engine:

    def __init__(self, params, benchamark, debug):
        self.logger = logging.getLogger('ATCV')

        self.frame_num = 0
        self.start_time = 0
        self.end_time = 0

        self.width = params['width']
        self.height = params['height']
        self.img_path = params['img_path']
        self.obj_path = params['obj_path']
        self.enable_undistorter = params['enable_undistorter']
        self.enable_beautifier = params['enable_beautifier']
        
        # TODO: Closeness and Center can change after undistortion
        diag = int(math.sqrt(self.width**2 + self.height**2))
        self.extractor = extractor.Extractor({
            'width': self.width,
            'height': self.height,
            'freq': SAMPLE_FREQ,
            'closeness': diag//40,
            'center': (self.width//2 - diag//40, self.height//5 * 2, diag//20, diag//20),
        }, benchamark, debug)

        if self.enable_undistorter:
            self.undistorter = undistortion.Undistortion(
                img_points_path=self.img_path, obj_points_path=self.obj_path
            )
        
        if self.enable_beautifier:
            self.beautifier = beautifier.Beautifier({}, debug)

        self.run_pre = []
        self.run_pre_sum = np.zeros((self.height, self.width, 3))

        self.run_post = []
        self.run_post_sum = np.zeros((self.height, self.width, 3))

        self.debug = debug
        self.show_image = debug

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

    '''
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
    '''
    def time(self):
        start_time = self.start_time
        end_time = time.time()
        c = 25
        if self.frame_num % c == 0:
            if start_time != 0:
                t = end_time - start_time
                fps = c / t
                self.logger.info('fps: {:.2f}'.format(fps))
            self.start_time = end_time
    
    def process(self, orig):
        src = orig.copy(order='C')
        
        # src, self.run_pre_sum = self.average(src, self.run_pre, self.run_pre_sum)

        if self.enable_undistorter:
            src = self.undistorter(src).copy(order='C')
        src = self.extractor(src, self.frame_num)
        if self.enable_beautifier:
            src = self.beautifier(src)
        src, self.run_post_sum = self.average(src, self.run_post, self.run_post_sum)
    
        if self.debug:
            self.time()
        self.frame_num += 1

        if self.show_image:
            show = np.hstack([orig, src])
            cv2.imshow('Video', show)
        return src
