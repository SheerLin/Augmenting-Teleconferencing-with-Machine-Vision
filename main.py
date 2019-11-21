#!/usr/bin/env python3

import fcntl
import sys
import os

import cv2

import engine
import undistortion
import interface

CAM_DEVICE_NUMBER = 0 # input device
CAP_DEVICE_NUMBER = 2 # output device
RESOLUTION = 1080

ENABLE_VIRTUAL_CAM = False
ENABLE_GUI = False

if ENABLE_VIRTUAL_CAM:
    import v4l2
'''
@brief Parses command line arguments.

@param args Command line arguments
@return cam_device_number, cap_device_number, resolution
'''
def parse_args(args):
    cam_device_number = CAM_DEVICE_NUMBER
    cap_device_number = CAP_DEVICE_NUMBER
    resolution = RESOLUTION
    if len(args) >= 2:
        cam_device_number = int(args[1])
    if len(args) >= 3:
        cap_device_number = int(args[2])
    if len(args) >= 4:
        resolution = int(args[3])
    return cam_device_number, cap_device_number, resolution

'''
@brief Returns width-height based on the resolution.
@return width, height
'''
def get_resolution(res):
    if res == 1080:
        width, height = 1920, 1080
    elif res == 720:
        width, height = 1280, 720
    elif res == 768:
        width, height = 1024, 768
    elif res == 600:
        width, height = 800, 600
    else: # 480p
        width, height = 640, 480
    return width, height

'''
@brief Obtains a video input device, and configures it for the given
       width-height. Reads first frame from the device to get the
       correct width-height.
       
@param dev_number Number of device to open
@param width Desired width
@param height Desired height
@return device, width, height
'''
def get_cam_device(dev_number, width, height):
    try:
        device = cv2.VideoCapture(dev_number)
        configure_cam_device(device, width, height)
        ret, im = device.read()
        if not ret:
            print("Can't read from device", dev_number)
            exit()
    except Exception as e:
        print("Exception in opening device", dev_number)
        print(e)
        exit()
    print("Old Width: {}, Height: {}".format(width, height))
    height = im.shape[0]
    width = im.shape[1]
    print("Cam Device: ", dev_number)
    print("Width: {}, Height: {}".format(width, height))
    return device, width, height


'''
@brief Obtains a video input device, using video file as input.
@param video_path Number of device to open
@return device, width, height
'''
def get_cam_device_from_video(video_path):
    try:
        device = cv2.VideoCapture(video_path)
        width = device.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        height = device.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        ret, im = device.read()
        if not ret:
            print("Can't read from video", video_path)
            exit()
    except Exception as e:
        print("Exception in opening video", video_path)
        print(e)
        exit()
    height = im.shape[0]
    width = im.shape[1]
    print("Video input: ", video_path)
    print("Width: {}, Height: {}".format(width, height))
    return device, width, height

'''
@brief Configures cam device for the given width-height.

@param device Cam device
@param width Desired width
@param height Desired height
@return Void
'''
def configure_cam_device(device, width, height):
    device.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    device.set(cv2.CAP_PROP_FRAME_HEIGHT, height)    

'''
@brief Obtains a video output device, and configure it for the given
       width-height.

@param dev_number Number of device to open
@param width Desired width
@param height Desired height
@return device
'''
def get_cap_device(dev_number, width, height):
    dev_name = '/dev/video' + str(dev_number)
    print("Capture Device: ", dev_name)
    try:
        device = open(dev_name, 'wb')
        if device is None:
            print("Bad capture device")
            exit()
    except Exception as e:
        print("Exception in opening device", dev_name)
        print(e)
        exit()
    configure_cap_device(device, width, height) 
    return device

'''
@brief Configures cap device for the given width-height, and output
       format.

@param device Cap device
@param width Desired width
@param height Desired height
@return Void
'''
def configure_cap_device(device, width, height):
    # get capabilities
    capability = v4l2.v4l2_capability()
    fcntl.ioctl(device, v4l2.VIDIOC_QUERYCAP, capability)
    print("v4l2 Driver: ", capability.driver)

    # set format 
    # https://linuxtv.org/downloads/v4l-dvb-apis/uapi/v4l/pixfmt.html
    format = v4l2.v4l2_format()
    format.type = v4l2.V4L2_BUF_TYPE_VIDEO_OUTPUT
    format.fmt.pix.pixelformat = v4l2.V4L2_PIX_FMT_BGR24
    format.fmt.pix.WIDTH = width
    format.fmt.pix.HEIGHT = height
    format.fmt.pix.field = v4l2.V4L2_FIELD_NONE
    format.fmt.pix.bytesperline = width * 2
    format.fmt.pix.sizeimage = width * height * 2
    format.fmt.pix.colorspace = v4l2.V4L2_COLORSPACE_JPEG
    fcntl.ioctl(device, v4l2.VIDIOC_S_FMT, format)

'''
@brief Main loop for reading the video stream from cam_device,
       processing, and writing the video stream to cap_device.

@param cam_device Cam device (input)
@param cap_device Cap device (output)
@param width Desired width
@param height Desired height
@return Void
'''
def process_video(cam_device, cap_device, width, height, img_path="", obj_path="", do_undistort=False):
    eng = engine.Engine(width, height, img_path, obj_path, do_undistort)

    while True:
        # try:
        ret, im = cam_device.read()
        if not ret:
            print("End of Input Stream")
            break
        
        out = eng.process(im)
        if ENABLE_VIRTUAL_CAM:
            cap_device.write(out)

        # except Exception as e:
        #     print(e)
        #     break
        
        # break on `escape` press
        if cv2.waitKey(1) == 27:
            break


if __name__== "__main__":

    # parse input
    # TODO
    # camera device, capture device, resoultion, enbale gui, 
    # enable vcam, distortion profile, video path
    CAM_DEVICE_NUMBER, CAP_DEVICE_NUMBER, RESOLUTION = parse_args(sys.argv)
    print("CAM_DEVICE_NUMBER", CAM_DEVICE_NUMBER)
    if ENABLE_VIRTUAL_CAM:
        print("CAP_DEVICE_NUMBER", CAP_DEVICE_NUMBER)
    print("RESOLUTION", RESOLUTION)
    
    # set up
    width, height = get_resolution(RESOLUTION)
    print(RESOLUTION, width, height)
    cam_device, width, height = get_cam_device(CAM_DEVICE_NUMBER, width, height)
    # cam_device, width, height = get_cam_device_from_video('data/wb_mengmeng.mov')
    # cam_device, width, height = get_cam_device_from_video('data/AccessMath_lecture_01_part_3.mp4')
    # cam_device, width, height = get_cam_device_from_video('raw-data/Piotr-wb.mov')
    # cam_device, width, height = get_cam_device_from_video('raw-data/classroom-wb.mov')
    # cam_device, width, height = get_cam_device_from_video('data/final4.webm')
    if ENABLE_VIRTUAL_CAM:
        cap_device = get_cap_device(CAP_DEVICE_NUMBER, width, height)
    else:
        cap_device = None

    # Initialize the profile path before processing video
    if ENABLE_GUI:
        # TODO
        # Start GUI for these arguments
        # camera device, capture device, resolution, enable gui,
        # enable vcam, distortion profile, video path

        undistortion_preprocessor = undistortion.UndistortionPreProcessor(CAM_DEVICE_NUMBER)
        device_to_profile = undistortion_preprocessor.init_profile_mapping()
        print("device_to_profile:", device_to_profile)

        # Should run process_video during the lifetime of user interface
        interface.initialize_ui(device_to_profile,cam_device, cap_device, width, height)

        print("Use UI")
        img_path = ""
        obj_path = ""
        chessboard_path = ""
        do_undistort = False

        pass

    else:
        # TODO - Should also accept chessboard path as parameter

        undistortion_preprocessor = undistortion.UndistortionPreProcessor(CAM_DEVICE_NUMBER)
        undistortion_preprocessor.init_profile_mapping()
        img_path, obj_path, do_undistort = undistortion_preprocessor()
        # print(img_path, obj_path)
        # print("do_undistort:",do_undistort)

        # do_undistort = True
        # img_path = "undistort/profiles/original4_img.npy"
        # obj_path = "undistort/profiles/original4_obj.npy"

        # process cap_device
        process_video(cam_device, cap_device, width, height, img_path, obj_path, do_undistort)

    # clean up
    del(cam_device)
    if ENABLE_VIRTUAL_CAM:
        cap_device.close()
