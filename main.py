#!/usr/bin/env python3

import fcntl
import sys
import os

import cv2
import v4l2

import engine

# video0 is the virtual camera capture device
# video1 is the virtual camera output device
# video2 is the integrated web cam
# video3 is the integrated web cam
# video4 is the integrated web cam
# video5 is the integrated web cam
# video6 is the usb cam
# video7 is the usb cam
# video8 is the usb cam
# video9 is the usb cam

CAM_DEVICE_NUMBER = 2 # input device
CAP_DEVICE_NUMBER = 0 # output device
HEIGHT, WIDTH = (480, 640) # frame size

def get_cam_device(dev_number):
    try:
        device = cv2.VideoCapture(dev_number)
        ret, im = device.read()
        if not ret:
            print("Can't read from device", dev_number)
            exit()
    except Exception as e:
        print("Warning: can't read from", dev_number)
        print(e)
        exit()
    return device, im.shape[0], im.shape[1]

def get_cap_device(dev_number):
    dev_name = '/dev/video' + str(dev_number)
    print("Capture Device: ", dev_name)
    try:
        device = open(dev_name, 'wb')
    except Exception as e:
        print("Exception in opening device", dev_name)
        print(e)
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
    eng = engine.Engine(WIDTH, HEIGHT, CAM_DEVICE_NUMBER)

    while True:
        try:
            ret, im = cam_device.read()
            
            if not ret:
                print("End of Input Stream")
                break
            out = eng.process(im)
            
            if cap_device is None:
                print("Bad Capture Device")
                break
            cap_device.write(out)

        except Exception as e:
            print(e)
            break
        
        # break on `escape` press
        if cv2.waitKey(1) == 27:
            break

def parse_args(args):
    cam_device_number = CAM_DEVICE_NUMBER
    cap_device_number = CAP_DEVICE_NUMBER
    if len(args) >= 2:
        cam_device_number = int(args[1])
    if len(args) >= 3:
        cap_device_number = int(args[2])
    return cam_device_number, cap_device_number


if __name__== "__main__":
    CAM_DEVICE_NUMBER, CAP_DEVICE_NUMBER = parse_args(sys.argv)
    print("CAM_DEVICE_NUMBER", CAM_DEVICE_NUMBER)
    print("CAP_DEVICE_NUMBER", CAP_DEVICE_NUMBER)
    
    cam_device, HEIGHT, WIDTH = get_cam_device(CAM_DEVICE_NUMBER)
    cap_device = get_cap_device(CAP_DEVICE_NUMBER)
    
    process_video(cam_device, cap_device)
    
    del(cam_device)
    cap_device.close()
