#!/usr/bin/env python3

import argparse
import fcntl
import sys
import os
import logging
import platform

import cv2

ENABLE_VIRTUAL_CAM = False
ENABLE_GUI = False

ENABLE_UNDISTORTER = False
ENABLE_BEAUTIFIER = True

BENCHMARK = False
DEBUG = False

import engine
import undistortion
import interface

CAM_DEVICE_NUMBER = 0 # input device
CAP_DEVICE_NUMBER = 0 # output device
RESOLUTION = 1080


FORMAT = '%(asctime)-15s %(name)s (%(levelname)s) > %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('ATCV')


def str2bool(x):
    return x.lower() in ('true')

'''
@brief Parses command line arguments.
@return args
'''
def parse_args():
    desc = 'Augmenting Tele-conferencing with Computer Vision'
    parser = argparse.ArgumentParser(description=desc)
    
    parser.add_argument('-i', '--inp', type=int, default=CAM_DEVICE_NUMBER, help='CAM_DEVICE_NUMBER: Camera input device (number, eg: 0)')
    parser.add_argument('-o', '--out', type=int, default=CAP_DEVICE_NUMBER, help='CAP_DEVICE_NUMBER: Virtual camera output device (number, eg: 1)')
    parser.add_argument('-r', '--res', type=int, default=RESOLUTION, help='RESOLUTION: Resolution for output (number, [1080, 720, 480, 768, 600])')
    parser.add_argument('-v', '--vcam', type=str2bool, default=ENABLE_VIRTUAL_CAM, help='ENABLE_VIRTUAL_CAM: Enable Virtual camera output (bool, [true, false])')
    parser.add_argument('-g', '--gui', type=str2bool, default=ENABLE_GUI, help='ENABLE_GUI: Enable GUI mode (bool, [true, false])')
    parser.add_argument('-ed', '--undistorter', type=str2bool, default=ENABLE_UNDISTORTER, help='ENABLE_UNDISTORTER: Enable undistorter component (bool, [true, false])')
    parser.add_argument('-eb', '--beautifier', type=str2bool, default=ENABLE_BEAUTIFIER, help='ENABLE_BEAUTIFIER: Enable beautifier component (bool, [true, false])')
    parser.add_argument('-b', '--benchmark', type=str2bool, default=BENCHMARK, help='BENCHAMRK: Enable benchmark mode (bool, [true, false])')
    parser.add_argument('-d', '--debug', type=str2bool, default=DEBUG, help='DEBUG: Enable debugging mode (bool, [true, false])')
    parser.add_argument('-p', '--profile', type=str, default=None, help='PROFILE: The profile name of undistorter when enabled (str, eg: "default")')

    args = parser.parse_args()
    return args

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
            print('Can\'t read from device', dev_number)
            exit()
    except Exception as e:
        print('Exception in opening device', dev_number)
        print(e)
        exit()
    height = im.shape[0]
    width = im.shape[1]
    logger.debug('Cam Device: {}'.format(dev_number))
    logger.debug('New Width: {}, Height: {}'.format(width, height))
    return device, width, height


'''
@brief Obtains a video input device, using video file as input.
@param video_path Path of video to open
@return device, width, height
'''
def get_cam_device_from_video(video_path):
    try:
        device = cv2.VideoCapture(video_path)
        ret, im = device.read()
        if not ret:
            print('Can\'t read from video', video_path)
            exit()
    except Exception as e:
        print('Exception in opening video', video_path)
        print(e)
        exit()
    height = im.shape[0]
    width = im.shape[1]
    logger.debug('Video Input: {}'.format(video_path))
    logger.debug('New Width: {}, Height: {}'.format(width, height))
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
    logger.debug('Capture Device: {}'.format(dev_name))
    try:
        device = open(dev_name, 'wb')
        if device is None:
            print('Bad capture device')
            exit()
    except Exception as e:
        print('Exception in opening device', dev_name)
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
    import v4l2
    capability = v4l2.v4l2_capability()
    fcntl.ioctl(device, v4l2.VIDIOC_QUERYCAP, capability)
    logger.debug('v4l2 Driver: {}'.format(capability.driver))

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
@brief Get width-height based on resolution. Configures cam and cap device 
       for the given width-height.
@param args Arguments dict from cmd line
@return cur_cam_device, cur_cap_device, cur_width, cur_height
'''
def configure_devices(args):
    cam_device_number = args['inp']
    cap_device_number = args['out']
    resolution = args['res']
    enable_virtual_cam = args['vcam']

    cur_width, cur_height = get_resolution(resolution)
    logger.debug('Width: {}, Height: {}'.format(cur_width, cur_height))
    cur_cam_device, cur_width, cur_height = get_cam_device(cam_device_number, cur_width, cur_height)
    logger.info('CAM_RESOLUTION: {}, {}'.format(cur_width, cur_height))
    if enable_virtual_cam:
        cur_cap_device = get_cap_device(cap_device_number, cur_width, cur_height)
    else:
        cur_cap_device = None

    return cur_cam_device, cur_cap_device, cur_width, cur_height

'''
@brief Main loop for reading the video stream from cam_device,
       processing, and writing the video stream to cap_device.
@param cam_device Cam device (input)
@param cap_device Cap device (output)
@param width Desired width
@param height Desired height
@param args Arguments dict from cmd line
@param img_path Parameter for undistorter
@param obj_path Parameter for undistorter
@param args dict
@return Void
'''
def process_video(cam_device, cap_device, width, height, img_path, obj_path, args):
    enable_virtual_cam = args['vcam']
    enable_undistorter = args['undistorter']
    enable_beautifier = args['beautifier']
    benchmark = args['benchmark']
    debug = args['debug']
    
    eng = engine.Engine({
        'width': width, 
        'height': height, 
        'img_path': img_path, 
        'obj_path': obj_path,
        'enable_undistorter': enable_undistorter,
        'enable_beautifier': enable_beautifier,
    }, benchmark, debug)

    while True:
        try:
            ret, im = cam_device.read()
            if not ret:
                logger.info('End of Input Stream')
                break
            
            out = eng.process(im)
            if enable_virtual_cam:
                cap_device.write(out)

        except Exception as e:
            logger.error(e)
        
        # Break on `escape` press
        if cv2.waitKey(1) == 27:
            break


if __name__== '__main__':
    # Parse arguments
    args = parse_args()
    cam_device_number = args.inp
    cap_device_number = args.out
    resolution = args.res
    enable_virtual_cam = args.vcam and platform.system() == "Linux"
    enable_gui = args.gui
    enable_undistorter = args.undistorter
    enable_beautifier = args.beautifier
    benchmark = args.benchmark
    debug = args.debug
    profile = args.profile
    
    log_level = logging.WARNING if not debug else logging.DEBUG
    logger.setLevel(log_level)
    logger.info('CAM_DEVICE_NUMBER: {}'.format(cam_device_number))
    if enable_virtual_cam:
        logger.info('CAP_DEVICE_NUMBER: {}'.format(cap_device_number))
    logger.info('RESOLUTION: {}'.format(resolution))
    if enable_undistorter and profile:
        logger.info('PROFILE: {}'.format(profile))

    cam_device = None
    cap_device = None
    
    # Initialize the undistorter profile path before processing video
    undistortion_preprocessor = undistortion.UndistortionPreProcessor(cam_device_number)
    profiles_map = undistortion_preprocessor.init_profile_mapping()
        
    if enable_gui:
        logger.info('Using GUI')

        interface.initialize_ui(profiles_map, args)

    else:
        logger.info('Using Cmd Line')
        
        cam_device, cap_device, width, height = configure_devices({
            'inp': cam_device_number,
            'out': cap_device_number,
            'res': resolution,
            'vcam': enable_virtual_cam,
        })

        img_path = None
        obj_path = None
        if enable_undistorter:
            img_path, obj_path, enable_undistorter = undistortion_preprocessor(profile)

        process_video(
            cam_device, cap_device,
            width, height,
            img_path, obj_path, 
            {
                'vcam': enable_virtual_cam,
                'undistorter': enable_undistorter,
                'beautifier': enable_beautifier,
                'benchmark': benchmark,
                'debug': debug,
            }
        )

    # Clean up
    if cam_device:
        del cam_device

    if cap_device:
        cap_device.close()
