#!/usr/bin/env python3

import fcntl
import sys
import os

import cv2
import v4l2

import engine

# video0 is the integrated web cam
CAM_DEVICE_NUMBER = 0
# CAM_DEVICE_NUMBER = 6

# video4 is the virtual camera capture device
CAP_DEVICE_NUMBER = 4

# frame size
# WIDTH = 1920
# HEIGHT = 1080
WIDTH, HEIGHT = (640, 480)

def get_cap_device(dev_number):
    dev_name = '/dev/video' + str(dev_number)
    print("Capture Device: ", dev_name)
    if not os.path.exists(dev_name):
        print("Warning: device does not exist", dev_name)
        exit()
    try:
        device = open(dev_name, 'w')
    except Exception:
        print("Exception in opening device")
        exit()
    configure_cap_device(device) 
    return device


def configure_cap_device(device):
    # get capabilities
    capability = v4l2.v4l2_capability()
    fcntl.ioctl(device, v4l2.VIDIOC_QUERYCAP, capability)
    print("v4l2 Driver: ", capability.driver)

    # set format 
    # https://linuxtv.org/downloads/v4l-dvb-apis/uapi/v4l/pixfmt.html
    format = v4l2.v4l2_format()
    format.type = v4l2.V4L2_BUF_TYPE_VIDEO_OUTPUT
    format.fmt.pix.pixelformat = v4l2.V4L2_PIX_FMT_BGR24
    format.fmt.pix.WIDTH = WIDTH
    format.fmt.pix.HEIGHT = HEIGHT
    format.fmt.pix.field = v4l2.V4L2_FIELD_NONE
    format.fmt.pix.bytesperline = WIDTH * 2
    format.fmt.pix.sizeimage = WIDTH * HEIGHT * 2
    format.fmt.pix.colorspace = v4l2.V4L2_COLORSPACE_JPEG
    fcntl.ioctl(device, v4l2.VIDIOC_S_FMT, format)


def process_video(cam_device, cap_device):
    eng = engine.Engine(WIDTH, HEIGHT)
    
    while True:
        try:
            ret, im = cam_device.read()
            if not ret:
                break
            out = eng.process(im)
            if cap_device is not None:
                cap_device.write(str(out.tostring()))
            # cv2.namedWindow('Orig', cv2.WINDOW_NORMAL)
            # cv2.imshow("Orig", im)

            # cv2.imshow("Video", out)
        except Exception as e:
            print(e)
            break
        
        # break on `escape` press
        if cv2.waitKey(1) == 27:
            break


if __name__== "__main__":
    if len(sys.argv) >= 2:
        CAP_DEVICE_NUMBER = sys.argv[1]
    
    cap_device = get_cap_device(CAP_DEVICE_NUMBER)
    cam_device = cv2.VideoCapture(CAM_DEVICE_NUMBER)
    
    process_video(cam_device, cap_device)
    
    del(cam_device)
    cap_device.close()
