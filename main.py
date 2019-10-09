import fcntl
import sys
import os

import cv2
import numpy as np
from v4l2 import *

import engine

# video0 is the integrated web cam
cam_device_number = 0

# video4 is the virtual camera capture device
cap_device_number = 4

# frame size
width = 640
height = 480

def get_cap_device(dev_number):
    dev_name = '/dev/video' + str(dev_number)
    print("Capture Device: " + dev_name)
    if not os.path.exists(dev_name):
        print("Warning: device does not exist", dev_name)
        exit()
    try:
        device = open(dev_name, 'w')
    except:
        print("Exception in opening device")
        exit()
    configure_cap_device(device) 
    return device


def configure_cap_device(device):
    # get capabilities
    capability = v4l2_capability()
    fcntl.ioctl(device, VIDIOC_QUERYCAP, capability)
    print("v4l2 Driver: " + capability.driver)

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


def process_video(cam_device, cap_device):
    eng = engine.Engine(width, height)
    
    while True:
        try:
            ret, im = cam_device.read()
            if not ret:
                continue
            out = eng.process(im)
            cap_device.write(out)
            cv2.imshow("Video", out)
        except Exception as e:
            print(e)
            break
        
        # break on `escape` press
        if cv2.waitKey(1) == 27:
            break


if __name__=="__main__":
    if len(sys.argv) >= 2:
        cap_device_number = sys.argv[1]
    
    cap_device = get_cap_device(cap_device_number)
    cam_device = cv2.VideoCapture(cam_device_number)
    
    process_video(cam_device, cap_device)
    
    del(cam_device)
    cap_device.close()
