#!/usr/bin/python3
import cv2
import glob
import os
import numpy as np
import subprocess
import shutil
import sys
import logging
import platform

profiles_folder = "undistort/profiles"
npy_file_postfix = ".npy"

default_chessboard_path = "undistort/data/chessboard/original/*"
default_chessboard_path2 = "undistort/data/chessboard/original4/*"

# Set default_img_points_path and default_obj_points_path
default_img_points_path = profiles_folder + "/default_img"
default_obj_points_path = profiles_folder + "/default_obj"

# For selecting profile: each line as <idVendor>:<idProduct>,<img_points_path>,<obj_points_path>
device_profile_mapping_file = profiles_folder + "/profile_mapping.txt"
DELIMITER = ","
default_profile_symbol = "d"
none_profile_symbol = "n"
logger = logging.getLogger("ATCV")


class Undistortion:
    def __init__(self, img_points_path=None, obj_points_path=None):
        self.logger = logging.getLogger("ATCV")

        self.img_points_path = img_points_path
        self.obj_points_path = obj_points_path

        # TO be initialized:
        # Arrays to store object points and image points from all the images.
        self.obj_points = []  # 3d point in real world space
        self.img_points = []  # 2d points in image plane.

        # debug
        self.show_image = True
        self.both_way = False
        self.crop = True
        self.imshow_size = cv2.WINDOW_NORMAL  # cv2.WINDOW_FULLSCREEN
        self.default_remap = False
        self.skip_undistort = False

        self.initialize()

    def __call__(self, img):
        if not self.skip_undistort:
            return self.calibrate_image(img)
        else:
            return img

    def initialize(self):
        """Initialize the needed info"""
        # Init self.obj_points and self.img_points
        if self.img_points_path and os.path.isfile(self.img_points_path) \
                and self.obj_points_path and os.path.exists(self.obj_points_path):
            pass
        else:
            self.skip_undistort = True
            self.logger.info("No available profile. Skip undistortion.")
            return False

        self.logger.info("Use img_points_path:{}".format(self.img_points_path))
        self.logger.info("Use obj_points_path:{}".format(self.obj_points_path))

        self.img_points, self.obj_points = Undistortion.deserialize(self.img_points_path, self.obj_points_path)

    @staticmethod
    def init_img_obj_points_from_chessboards(init_chessboard_path):
        """Get image points and object points from the pictures in the chessboard_path"""
        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6 * 7, 3), np.float32)
        objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

        images = glob.glob(init_chessboard_path)
        valid_pics = 0
        obj_points = []
        img_points = []

        if len(images) == 0:
            # Add asterisk manually and retry
            if "*" not in init_chessboard_path:
                if init_chessboard_path.endswith("/"):
                    init_chessboard_path += "*"
                else:
                    init_chessboard_path += "/*"

                images = glob.glob(init_chessboard_path)

            if len(images) == 0:
                logger.error("NO pictures under chessboard path:{}".format(init_chessboard_path))
                logger.error("Chessboard path should be in the format of:<relative path>/*")
                exit()

        for file_name in images:
            img = cv2.imread(file_name)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)

            # If found, add object points, image points (after refining them)
            if ret:
                obj_points.append(objp)

                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                img_points.append(corners2)

                valid_pics += 1
                logger.debug("Initialized from {} pictures:{}".format(valid_pics, file_name))
            else:
                logger.debug("Failed to findChessboardCorners. Skip current image:{}".format(file_name))
                os.remove(file_name)

        cv2.destroyAllWindows()
        logger.debug("Collect 3d point from {} pictures.".format(valid_pics))

        return img_points, obj_points

    @staticmethod
    def chessboard_path_to_profile(cur_profile_name, set_up_chessboard_path, img_points_path, obj_points_path,
                                   devices: list):
        """1. Save the points from chessboard path pictures to img_points_path, obj_points_path
        2. Take devices list and add the mapping to device_profile_mapping_file"""
        img_points, obj_points = Undistortion.init_img_obj_points_from_chessboards(set_up_chessboard_path)
        Undistortion.serialize(img_points, obj_points, img_points_path, obj_points_path)

        if devices and len(devices) > 0:
            Undistortion.save_device_to_profile(devices, img_points_path, obj_points_path, cur_profile_name)

    @staticmethod
    def save_device_to_profile(devices: list, img_points_path, obj_points_path, cur_profile_name: str):
        """Write to device_profile_mapping_file file"""

        logger.debug("In save_device_to_profile")
        logger.debug("devices: {}", devices)
        logger.debug("img_points_path: {}", img_points_path)
        logger.debug("obj_points_path: {}", obj_points_path)
        logger.debug("device_profile_mapping_file: {}", device_profile_mapping_file)
        logger.debug("cur_profile_name: {}", cur_profile_name)

        if device_profile_mapping_file:
            # 1. Set the open mode
            mode = 'a+'
            if not os.path.exists(device_profile_mapping_file):
                mode = 'w+'
            elif not os.path.isfile(device_profile_mapping_file):
                shutil.rmtree(device_profile_mapping_file)
                mode = 'w+'
                pass

            # 2. Write to file in the format of <device>,<img_points_path>,<obj_points_path>
            with open(device_profile_mapping_file, mode) as mapping_file:
                for cur_device in devices:
                    line = cur_profile_name + DELIMITER + cur_device + DELIMITER + \
                           img_points_path + DELIMITER + obj_points_path + "\n"
                    logger.debug(line)
                    mapping_file.write(line)

        else:
            logger.error("Should define mapping file path variable 'device_profile_mapping_file' in undistortion.py")

        pass

    @staticmethod
    def serialize(img_points, obj_points, img_path_no_post_fix, obj_path_no_post_fix):

        img_points_np = np.asarray(img_points)
        obj_points_np = np.asarray(obj_points)

        np.save(img_path_no_post_fix, img_points_np)
        np.save(obj_path_no_post_fix, obj_points_np)

        return img_points, obj_points

    @staticmethod
    def deserialize(de_img_path, de_obj_path):
        if not de_img_path or not os.path.exists(de_img_path):
            logger.error("img_path({}):File not exist".format(de_img_path))
            return None, None

        if not de_obj_path or not os.path.exists(de_obj_path):
            logger.error("obj_path({}):File not exist".format(de_obj_path))
            return None, None

        img_points = np.load(de_img_path)
        obj_points = np.load(de_obj_path)

        return img_points, obj_points

    @staticmethod
    def check_np_array_equals(original, deserialized):
        index = 0
        for deserialized_points in deserialized:
            deserialized_np = np.array(deserialized_points)
            original_point_np = original[index]
            index += 1
            print(np.equal(deserialized_np, original_point_np))

    def calibrate_image(self, image):
        """Input a single frame and return the frame after undistortion"""
        if self.obj_points is None or self.img_points is None:
            self.logger.debug("No self.obj_points or no self.img_points")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.obj_points, self.img_points,
                                                           gray.shape[::-1], None, None)

        h, w = image.shape[:2]
        new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        if self.show_image:
            cv2.namedWindow("Before undistortion", self.imshow_size)
            try:
                cv2.imshow("Before undistortion", image)
            except Exception as e:
                self.logger.error("Display(Before undistortion) error:{}".format(e))

        if not self.default_remap:
            dst = cv2.undistort(image, mtx, dist, None, new_camera_mtx)
        else:
            mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, new_camera_mtx, (w, h), 5)
            dst = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)

        if self.show_image:
            cv2.namedWindow("After undistortion", self.imshow_size)
            try:
                cv2.imshow("After undistortion", dst)
            except Exception as e:
                self.logger.error("Display(After undistortion) error:{}".format(e))

        if self.crop:
            # crop the image
            x, y, w, h = roi
            dst = dst[y:y + h, x:x + w]
            if self.show_image:
                cv2.namedWindow("crop", self.imshow_size)
                try:
                    cv2.imshow("crop", dst)
                except Exception as e:
                    self.logger.error("Display(crop) error:{}".format(e))

        return dst


class UndistortionPreProcessor:
    def __init__(self, cam_device_number: int = 0):
        self.device_to_profile = dict()  # Profile for selecting profile: <device> -> [list of (img path,obj path)]
        self.cam_device_number = cam_device_number
        self.logger = logging.getLogger("ATCV")

        # DEBUG
        self.find_all_device = True

    def __call__(self, profile: str):
        return self.__select_profile(profile)

    def init_profile_mapping(self):
        """Reading from mapping profile to initialize self.device_to_profile
        and initialize the device name"""

        if device_profile_mapping_file and os.path.isfile(device_profile_mapping_file):

            with open(device_profile_mapping_file, 'r+') as file:
                lines = file.readlines()
                for line in lines:
                    line_array = line.strip('\n').split(DELIMITER)
                    if len(line_array) != 4:
                        self.logger.debug("Skip current line due to incorrect format: {}".format(line_array))
                        continue

                    cur_profile_name = str(line_array[0])
                    current_device = str(line_array[1])
                    current_img_path = str(line_array[2])
                    current_obj_path = str(line_array[3])

                    if not self.device_to_profile:
                        self.device_to_profile = dict()

                    if current_device not in self.device_to_profile:
                        self.device_to_profile[current_device] = list()

                    self.device_to_profile[current_device].append(
                        (cur_profile_name, current_img_path, current_obj_path))

            self.logger.debug("After initialize self.device_to_profile: {}".format(self.device_to_profile))

        return self.device_to_profile

    # TODO
    def __select_profile(self, profile: str):
        """
        Function:
            1. Find current usb devices and check if it matches with the existing mapping
            2. Find the corresponding profile according to devices, if multiple available devices found
            Let user select with device(profile) to be used

        Returns:
            Path to img points npy file (with post fix)
            Path to obj points npy file (with post fix)
        """
        self.device_name = UndistortionPreProcessor.find_device_id_by_cam_device_number(self.cam_device_number)

        enable_undistorter = True

        self.logger.info("Current device:{}".format(self.device_name))

        if len(self.device_to_profile.keys()) > 0:

            # 1. Find out list of devices
            # TODO[fusion]  - uncomment this
            list_of_devices = UndistortionPreProcessor.get_usb_devices()

            # TODO[fusion]  - remove this
            # list_of_devices = ['05a3:9230', '046d:0837']

            # 2. Iterate through the list and find if the current device has profile
            available_profiles_map = dict()
            for current_device in list_of_devices:
                if current_device in self.device_to_profile:

                    if self.find_all_device or not self.device_name or current_device == self.device_name:
                        if current_device not in available_profiles_map:
                            available_profiles_map[current_device] = set()

                        for profile_tuple in self.device_to_profile[current_device]:
                            tmp_img_path = str(profile_tuple[1])
                            tmp_obj_path = str(profile_tuple[2])
                            if tmp_img_path and os.path.exists(tmp_img_path + npy_file_postfix) \
                                    and tmp_obj_path and os.path.exists(tmp_obj_path + npy_file_postfix):
                                available_profiles_map[current_device].add(profile_tuple)

            # 3. If there is no  available profile pairs, pass
            if len(available_profiles_map) == 0:
                self.logger.info("Skip select profile: No available device in profile mapping")
                pass

            # 4. If there is only one available profile pair, return it
            no_profile = False
            if len(available_profiles_map) == 1:

                for the_only_device, the_tuples in available_profiles_map.items():
                    if len(the_tuples) == 0:
                        self.logger.debug("Skip select profile: No available profile path found for device {}",
                                          the_only_device)
                        no_profile = True
                        break

                    elif len(the_tuples) == 1:
                        self.logger.debug("Single profile pair found!!")
                        selected_img_path = str(list(the_tuples)[0][1])
                        selected_obj_path = str(list(the_tuples)[0][2])
                        return selected_img_path + npy_file_postfix, selected_obj_path + npy_file_postfix, \
                               enable_undistorter

            # 5. If there are multiple available profile pairs, let user select
            if not no_profile:
                id_to_profile = UndistortionPreProcessor.profile_map_formatter(available_profiles_map)
                while True:
                    user_selected = input()
                    if user_selected and len(user_selected) > 0 and str(user_selected).isdigit() \
                            and int(user_selected) in id_to_profile:
                        self.logger.debug("Valid Input:{} {}".format(user_selected, id_to_profile[int(user_selected)]))
                        selected_img_path = str(id_to_profile[int(user_selected)][1])
                        selected_obj_path = str(id_to_profile[int(user_selected)][2])
                        return selected_img_path + npy_file_postfix, selected_obj_path + npy_file_postfix, \
                               enable_undistorter

                    # Enable default profiling
                    if str(user_selected).lower() == default_profile_symbol:
                        break

                    #  Enable skipping undistortion
                    elif str(user_selected).lower() == none_profile_symbol:
                        enable_undistorter = False
                        self.logger.info("Skipping undistortion!")
                        break

                    else:
                        self.logger.error("Invalid input: {}", user_selected)

        if default_img_points_path \
                and os.path.isfile(default_img_points_path + npy_file_postfix) \
                and default_obj_points_path \
                and os.path.exists(default_obj_points_path + npy_file_postfix):
            return default_img_points_path + npy_file_postfix, default_obj_points_path + npy_file_postfix, enable_undistorter

        else:
            return False, False, enable_undistorter

    @staticmethod
    def find_device_id_by_cam_device_number(cam_device_number):
        device_vendor_product = None

        # Find the vendor id and product id of device
        driver_path = "/sys/class/video4linux/video" + str(cam_device_number) + "/device/input/"
        if not os.path.exists(driver_path):
            logger.debug("Driver folder not exist: {}".format(driver_path))
            return device_vendor_product

        input_folder = ""
        try:
            input_folder = os.listdir(driver_path)[0]
        except Exception as e:
            logger.error("Exception when finding folder({}): {}".format(driver_path + input_folder, e))
            return device_vendor_product

        driver_path += (input_folder + "/id")
        if not os.path.exists(driver_path):
            logger.error("Driver input id folder not exist: {}".format(driver_path))
            return device_vendor_product

        # Check vendor id
        vendor_path = driver_path + "/vendor"
        if not os.path.exists(vendor_path):
            logger.error("Vendor file not exist: {}".format(vendor_path))
            return device_vendor_product

        id_vendor = ""
        with open(vendor_path, 'r') as f:
            id_vendor = str(f.readline()).strip()

        # Check product id
        product_path = driver_path + "/product"
        if not os.path.exists(product_path):
            logger.error("Product file not exist: {}".format(product_path))
            return device_vendor_product

        id_product = ""
        with open(product_path, 'r') as f:
            id_product = str(f.readline()).strip()

        device_vendor_product = id_vendor + ":" + id_product

        return device_vendor_product

    @staticmethod
    def get_usb_devices():
        """Get the list of <idVendor>:<idProduct> from cmd lsusb"""
        # TODO - MAC compatibility
        cmd = "lsusb | awk '{print $6 }'"
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = p.communicate()[0].decode("utf-8")
        usb_devices = str(output).splitlines()
        return usb_devices

    @staticmethod
    def profile_map_formatter(profile_map: dict):
        """
        Function:
            Give each profile pair a unique id and print them in user readable format

        Returns:
            A dictionary: id -> (img path, obj path)
        """
        counter = 0
        padding = "    "
        id_to_profile_pair = dict()
        print("Please select a profile by entering the #:\n"
              "(or \"" + default_profile_symbol + "\" for default profile, "
                                                  "\"" + none_profile_symbol + "\" for not to undistort)")
        for cur_device, cur_tuple_list_or_set in profile_map.items():
            cur_tuple_list = list(cur_tuple_list_or_set)
            print(padding + UndistortionPreProcessor.get_usb_device(device_id=cur_device))

            for cur_tuple in cur_tuple_list:
                print(padding + padding, counter, "=", cur_tuple[0])
                id_to_profile_pair[counter] = cur_tuple
                counter += 1

        return id_to_profile_pair

    @staticmethod
    def get_usb_device(device_id):
        """For Linux, get the (first) line in lsusb for device with device_id. Otherwise just return device_id"""
        if platform.system() == "Linux":
            cmd = "lsusb | grep " + str(device_id)
            p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            output = p.communicate()[0].decode("utf-8")
            usb_devices = str(output).splitlines()
            if not usb_devices or len(usb_devices) == 0:
                return []
            else:
                return usb_devices[0]
        else:
            return device_id

    @staticmethod
    def get_videos_list():
        result = list()

        if platform.system() == "Linux":
            cmd = "ls /dev/video*"
            p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            output = p.communicate()[0].decode("utf-8")
            raw_list = str(output).splitlines()
            for cur_video in raw_list:
                cur_video = cur_video.strip().split("/")[2]
                result.append(cur_video)
        else:
            result.append("video0")

        return result


def get_default_profile_pair():
    return (default_img_points_path, default_obj_points_path)


def parse_args(args):
    if len(args) < 6:
        usage()
        exit()

    cur_profile_name = str(args[1])

    # Check chessboard path
    cur_chessboard_path = str(args[2])
    images = glob.glob(cur_chessboard_path)

    if len(images) == 0:
        # Add asterisk manually and retry
        if "*" not in cur_chessboard_path:
            if cur_chessboard_path.endswith("/"):
                cur_chessboard_path += "*"
            else:
                cur_chessboard_path += "/*"

            images = glob.glob(cur_chessboard_path)

        if len(images) == 0:
            logger.error("NO pictures under chessboard path:{}".format(cur_chessboard_path))
            logger.error("Chessboard path should be in the format of:<relative path>/*")
            exit()

    cur_img_path = str(args[3])
    cur_obj_path = str(args[4])
    cur_device_list = set()

    index = 0
    while index < len(args):
        if index > 4:
            cur_device_list.add(str(args[index]))
        index += 1

    return cur_profile_name, cur_chessboard_path, cur_img_path, cur_obj_path, list(cur_device_list)


def usage():
    print("Usage: python3 undistortion.py <profile name> <chessboard path> <img point path> "
          "<obj point path> <device1> [<device2> ... <device n>]")
    test_profile_name = "slight_640_800"
    test_img_points_path = "undistort/profiles/img1"
    test_obj_points_path = "undistort/profiles/obj1"
    print("   <profile name>   : The name of the profile, e.g." + test_profile_name)
    print("   <chessboard path>: Path to the folder of chessboard pictures, e.g.\"" +
          default_chessboard_path2 + "\"")
    print("   <img point path> : Path to save the img points without post fix, e.g.\"" +
          test_img_points_path + "\"")
    print("   <obj point path> : Path to save the obj points without post fix, e.g.\"" +
          test_obj_points_path + "\"")
    print("   <device n>       : Device in the format of <idVendor>:<idProduct>, e.g.05a3:9230")


if __name__ == "__main__":
    # Main function is for build profile for a list of devices
    profile_name, chessboard_path, img_path, obj_path, device_list = parse_args(sys.argv)
    logger.debug("profile_name: {}".format(profile_name))
    logger.debug("chessboard_path: {}".format(chessboard_path))
    logger.debug("img_path: {}".format(img_path))
    logger.debug("obj_path: {}".format(obj_path))
    logger.debug("device_list: {}".format(device_list))
    Undistortion.chessboard_path_to_profile(profile_name, chessboard_path, img_path, obj_path, device_list)

    # Example:
    # Undistortion.chessboard_path_to_profile(
    #     default_chessboard_path2, "undistort/profiles/test_img1",
    #     "undistort/profiles/test_obj1", ["05a3:9230", "test"]
    # )
