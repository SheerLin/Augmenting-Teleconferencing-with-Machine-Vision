import fcntl, sys, os
from v4l2 import *
import time
import cv2
import math
import numpy as np

# video0 is the integrated web cam
camera_port = 0 

# video4 is the virtual camera cature device
dev_name = '/dev/video4'

# frame size
width = 640
height = 480
    
if __name__=="__main__":
    if len(sys.argv) >= 2:
        dev_name = sys.argv[1]
    if not os.path.exists(dev_name):
        print("Warning: device does not exist", dev_name)
    device = open(dev_name, 'wr')

    # get capabilities
    capability = v4l2_capability()
    fcntl.ioctl(device, VIDIOC_QUERYCAP, capability)
    print("v4l2 driver: " + capability.driver)

    # set format 
    # https://linuxtv.org/downloads/v4l-dvb-apis/uapi/v4l/pixfmt.html
    format = v4l2_format()
    format.type = V4L2_BUF_TYPE_VIDEO_OUTPUT
    format.fmt.pix.pixelformat = V4L2_PIX_FMT_BGR24
    format.fmt.pix.width = width
    format.fmt.pix.height = height
    format.fmt.pix.field = V4L2_FIELD_NONE
    format.fmt.pix.bytesperline = width * 2
    format.fmt.pix.sizeimage = width * height * 2
    format.fmt.pix.colorspace = V4L2_COLORSPACE_JPEG
    fcntl.ioctl(device, VIDIOC_S_FMT, format)
    
    camera = cv2.VideoCapture(camera_port)
    while True:
        ret, im = camera.read()
        
        # https://docs.opencv.org/3.4/d8/d01/group__imgproc__color__conversions.html
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        
        cv2.imshow("video", im)                
        device.write(im)

        if cv2.waitKey(1) == 27:
            break
    
    del(camera)
    device.close()
