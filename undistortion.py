#!/usr/bin/python3
import cv2
import glob
import os
import time
import numpy as np
import subprocess
import shutil
import sys

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


# General steps
#  1. If has chessboards images folder path, use this path to get point.
#  2. if has profile path for image points and obj_points, use this profile
#  3. If no profile path and no has chessboards images folder path, select from the existing profiles
#  4. Undistort the image
class Undistortion:
    def __init__(self, chessboard_folder_path=None, img_points_path=None,obj_points_path=None, ):
        # TODO[low] - Refactor the format of parameters
        self.img_points_path = img_points_path
        self.obj_points_path = obj_points_path
        self.chessboard_folder_path = chessboard_folder_path
        # self.cam_device_number = cam_device_number

        # TO be initialized:
        # Arrays to store object points and image points from all the images.
        self.obj_points = []  # 3d point in real world space
        self.img_points = []  # 2d points in image plane.
        # self.device_to_profile = dict()  # Profile for selecting profile: <device> -> [list of (img path,obj path)]
        # self.device_name = None

        # debug
        self.show_image = False
        self.save_image = False
        self.both_way = False
        self.crop = True
        self.imshow_size = cv2.WINDOW_NORMAL  # cv2.WINDOW_FULLSCREEN
        self.default_remap = False
        # self.find_all_device = False
        self.skip_undistort = False

        self.initialize()

    def __call__(self, img):
        if not self.skip_undistort:
            return self.calibrate_image(img)
        else:
            return img

    def initialize(self):
        """Initialize the needed info"""

        #  TODO - remove 1. Init self.device_to_profile
        # self.__init_profile_mapping()

        # print("self.chessboard_folder_path", self.chessboard_folder_path)
        # print("self.img_points_path", self.img_points_path)
        # print("self.obj_points_path", self.obj_points_path)

        # 2. Init self.obj_points and self. img_points
        # if self.chessboard_folder_path and os.path.exists(self.chessboard_folder_path):
        if self.chessboard_folder_path:
            print("Use input chessboard folder path:", self.chessboard_folder_path)
            self.img_points, self.obj_points = Undistortion.init_img_obj_points_from_chessboards(
                self.chessboard_folder_path)
        else:
            if self.img_points_path and os.path.isfile(self.img_points_path) \
                    and self.obj_points_path and os.path.exists(self.obj_points_path):
                pass
            else:
                # TODO - return false here
                self.skip_undistort = True
                print("No available profile. Skip undistortion.")
                return False

                # self.img_points_path, self.obj_points_path = self.__select_profile()
                # if not self.img_points_path or not self.obj_points_path:
                #     print("Failed to initialize profile path.")
                #     return False

            print("Use img_points_path:", self.img_points_path)
            print("Use obj_points_path:", self.obj_points_path)

            self.img_points, self.obj_points = Undistortion.deserialize(self.img_points_path, self.obj_points_path)

    @staticmethod
    def init_img_obj_points_from_chessboards(init_chessboard_path):
        """Get image points and object points from the pictures in the chessboard_path"""
        start_time = time.time()
        print("in __init_img_obj_points_from_chessboards")

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
            # TODO[low] - add asterisk manually
            print("NO pictures under chessboard path:", init_chessboard_path)
            print("Chessboard path should be in the format of:<relative path>/*")
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

                # Draw and display the corners
                # img = cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
                # cv2.imshow('img', img)
                # cv2.waitKey(100)

                valid_pics += 1
                print("Initialized from ", valid_pics, "pictures:", file_name)
            else:
                print("Failed to findChessboardCorners. Skip current image:", file_name)
                os.remove(file_name)

        elapsed_time = time.time() - start_time
        cv2.destroyAllWindows()
        print("Collect 3d point from ", valid_pics, " pictures:", elapsed_time)
        # print("img_points", type(img_points), "len=", len(img_points))
        # print("obj_points", type(obj_points), "len=", len(obj_points))

        return img_points, obj_points

    @staticmethod
    def chessboard_path_to_profile(set_up_chessboard_path, img_points_path, obj_points_path, devices: list):
        """1. Save the points from chessboard path pictures to img_points_path, obj_points_path
        2. Take devices list and add the mapping to device_profile_mapping_file"""
        img_points, obj_points = Undistortion.init_img_obj_points_from_chessboards(set_up_chessboard_path)
        Undistortion.serialize(img_points, obj_points, img_points_path, obj_points_path)

        if devices and len(devices) > 0:
            Undistortion.save_device_to_profile(devices, img_points_path, obj_points_path)

    @staticmethod
    def save_device_to_profile(devices: list, img_points_path, obj_points_path):
        """Write to device_profile_mapping_file file"""

        print("In define_para_to_profile:")
        print("devices:", devices)
        print("img_points_path:", img_points_path)
        print("obj_points_path:", obj_points_path)
        print("device_profile_mapping_file:", device_profile_mapping_file)

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
                    line = cur_device + DELIMITER + img_points_path + DELIMITER + obj_points_path + "\n"
                    print(line)
                    mapping_file.write(line)

        else:
            print("ERROR: Should define mapping file path variable 'device_profile_mapping_file' in undistortion.py")

        pass

    @staticmethod
    def serialize(img_points, obj_points, img_path_no_post_fix, obj_path_no_post_fix):

        img_points_np = np.asarray(img_points)
        # print("img_points_np shape:", img_points_np.shape)
        obj_points_np = np.asarray(obj_points)
        # print("obj_points_np shape:", obj_points_np.shape)

        np.save(img_path_no_post_fix, img_points_np)
        np.save(obj_path_no_post_fix, obj_points_np)

        return img_points, obj_points

    @staticmethod
    def deserialize(de_img_path, de_obj_path):
        if not de_img_path or not os.path.exists(de_img_path):
            print("img_path(", de_img_path + "):File not exist")
            return None, None

        if not de_obj_path or not os.path.exists(de_obj_path):
            print("obj_path(", de_obj_path + "):File not exist")
            return None, None

        img_points = np.load(de_img_path)
        obj_points = np.load(de_obj_path)

        return img_points, obj_points

    @staticmethod
    def check_np_array_equals(original, deserialized):
        index = 0
        # result = True
        for deserialized_points in deserialized:
            deserialized_np = np.array(deserialized_points)
            original_point_np = original[index]
            index += 1
            print(np.equal(deserialized_np, original_point_np))
            # result = result and np.equal(deserialized_np, original_point_np)

        # return result

    def calibrate_image(self, image):
        """Input a single frame and return the frame after undistortion"""
        if self.obj_points is None or self.img_points is None:
            print("not self.obj_points or not self.img_points")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.obj_points, self.img_points,
                                                           gray.shape[::-1], None, None)

        h, w = image.shape[:2]
        new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

        if self.show_image:
            cv2.namedWindow("before undistortion", self.imshow_size)
            cv2.imshow("before undistortion", image)

        if not self.default_remap:
            dst = cv2.undistort(image, mtx, dist, None, new_camera_mtx)
        else:
            mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, new_camera_mtx, (w, h), 5)
            dst = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)

        if self.show_image:
            cv2.namedWindow("after undistortion", self.imshow_size)
            cv2.imshow("after undistortion", dst)

        if self.crop:
            # crop the image
            x, y, w, h = roi
            dst = dst[y:y + h, x:x + w]
            if self.show_image:
                cv2.namedWindow("crop", self.imshow_size)
                cv2.imshow("crop", dst)

        return dst


class UndistortionPreProcessor:
    def __init__(self, cam_device_number: int = 0):
        self.device_to_profile = dict()  # Profile for selecting profile: <device> -> [list of (img path,obj path)]
        self.cam_device_number = cam_device_number

        # DEBUG
        self.find_all_device = False

        self.__init_profile_mapping()

    def __call__(self):
        return self.__select_profile()

    def __init_profile_mapping(self):
        """Reading from mapping profile to initialize self.device_to_profile
        and initialize the device name"""

        if device_profile_mapping_file and os.path.isfile(device_profile_mapping_file):

            with open(device_profile_mapping_file, 'r+') as file:
                lines = file.readlines()
                for line in lines:
                    line_array = line.strip('\n').split(",")
                    if len(line_array) != 3:
                        # print("Skip current line due to incorrect format:", line_array)
                        continue

                    current_device = str(line_array[0])
                    current_img_path = str(line_array[1])
                    current_obj_path = str(line_array[2])

                    if not self.device_to_profile:
                        self.device_to_profile = dict()

                    if current_device not in self.device_to_profile:
                        self.device_to_profile[current_device] = list()

                    self.device_to_profile[current_device].append((current_img_path, current_obj_path))

            print("After initialize self.device_to_profile:", self.device_to_profile)

        return

    def __select_profile(self):
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
        print("Current device:", self.device_name)

        if len(self.device_to_profile.keys()) > 0:

            # 1. Find out list of devices
            list_of_devices = UndistortionPreProcessor.get_usb_devices()

            # 2. Iterate through the list and find if the current device has profile
            # print(list_of_devices)
            available_profiles_map = dict()
            for current_device in list_of_devices:
                if current_device in self.device_to_profile:

                    if self.find_all_device or not self.device_name or current_device == self.device_name:
                        if current_device not in available_profiles_map:
                            available_profiles_map[current_device] = set()

                        for profile_pair in self.device_to_profile[current_device]:
                            tmp_img_path = str(profile_pair[0])
                            tmp_obj_path = str(profile_pair[1])
                            if tmp_img_path and os.path.exists(tmp_img_path + npy_file_postfix) \
                                    and tmp_obj_path and os.path.exists(tmp_obj_path + npy_file_postfix):
                                available_profiles_map[current_device].add(profile_pair)

            # 3. If there is no  available profile pairs, pass
            if len(available_profiles_map) == 0:
                print("Skip select profile: No available device in profile mapping")
                pass

            # 4. If there is only one available profile pair, return it
            no_profile = False
            if len(available_profiles_map) == 1:

                for the_only_device, the_pairs in available_profiles_map.items():
                    if len(the_pairs) == 0:
                        print("Skip select profile: No available profile path found for device", the_only_device)
                        no_profile = True
                        break
                    elif len(the_pairs) == 1:
                        print("Single profile pair found!!")
                        selected_img_path = str(list(the_pairs)[0][0])
                        selected_obj_path = str(list(the_pairs)[0][1])
                        return selected_img_path + npy_file_postfix, selected_obj_path + npy_file_postfix

            # 5. If there are multiple available profile pairs, let user select
            if not no_profile:
                id_to_profile = UndistortionPreProcessor.profile_map_formatter(available_profiles_map,
                                                                               "available_profiles_map")
                while True:
                    user_selected = input()
                    if user_selected and len(user_selected) > 0 and str(user_selected).isdigit() \
                            and int(user_selected) in id_to_profile:
                        print("Valid Input:", int(user_selected), id_to_profile[int(user_selected)])
                        selected_img_path = str(id_to_profile[int(user_selected)][0])
                        selected_obj_path = str(id_to_profile[int(user_selected)][1])
                        return selected_img_path + npy_file_postfix, selected_obj_path + npy_file_postfix

                    # Enable default profiling
                    if str(user_selected).lower() == default_profile_symbol:
                        break

                    #  Enable skipping undistortion
                    elif str(user_selected).lower() == none_profile_symbol:
                        self.skip_undistort = True
                        print("Skipping undistortion!")
                        break

                    else:
                        print("Invalid input:", user_selected)

        if default_img_points_path \
                and os.path.isfile(default_img_points_path + npy_file_postfix) \
                and default_obj_points_path \
                and os.path.exists(default_obj_points_path + npy_file_postfix):
            return default_img_points_path + npy_file_postfix, default_obj_points_path + npy_file_postfix

        else:
            return False, False

    @staticmethod
    def find_device_id_by_cam_device_number(cam_device_number):
        device_vendor_product = None

        # 1. Find the name of device: optional
        name_path = "/sys/class/video4linux/video" + str(cam_device_number) + "/name"
        if not os.path.exists(name_path):
            print("Name file not exist:", name_path)
            return device_vendor_product

        with open(name_path, 'r') as f:
            name = f.readline()
            print("Name:", name)

        # 2. Find the vendor id and product id of device
        driver_path = "/sys/class/video4linux/video" + str(cam_device_number) + "/device/input/"
        if not os.path.exists(driver_path):
            print("Driver folder not exist:", driver_path)
            return device_vendor_product

        input_folder = ""
        try:
            input_folder = os.listdir(driver_path)[0]
        except:
            print("Driver input folder not exist:", driver_path + input_folder)
            return device_vendor_product

        driver_path += (input_folder + "/id")
        if not os.path.exists(driver_path):
            print("Driver input id folder not exist:", driver_path)
            return device_vendor_product

        # Check vendor id
        vendor_path = driver_path + "/vendor"
        if not os.path.exists(vendor_path):
            print("Vendor file not exist:", vendor_path)
            return device_vendor_product

        id_vendor = ""
        with open(vendor_path, 'r') as f:
            id_vendor = str(f.readline()).strip()

        # Check product id
        product_path = driver_path + "/product"
        if not os.path.exists(product_path):
            print("Product file not exist:", product_path)
            return device_vendor_product

        id_product = ""
        with open(product_path, 'r') as f:
            id_product = str(f.readline()).strip()

        device_vendor_product = id_vendor + ":" + id_product

        return device_vendor_product

    @staticmethod
    def get_usb_devices():
        """Get the list of <idVendor>:<idProduct> from cmd lsusb"""
        cmd = "lsusb | awk '{print $6 }'"
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = p.communicate()[0].decode("utf-8")
        usb_devices = str(output).splitlines()
        return usb_devices

    @staticmethod
    def profile_map_formatter(profile_map: dict, name: str):
        """
        Function:
            Give each profile pair a unique id and print them in user readable format

        Returns:
            A dictionary: id -> (img path, obj path)
        """
        counter = 0
        padding = "    "
        id_to_profile_pair = dict()
        # print(name, ":")
        print("Please select a profile by entering the #:\n"
              "(or \"" + default_profile_symbol + "\" for default profile, "
                                                  "\"" + none_profile_symbol + "\" for not to undistort)")
        for cur_device, cur_pair_list_or_set in profile_map.items():
            cur_pair_list = list(cur_pair_list_or_set)
            print(padding + UndistortionPreProcessor.get_usb_device(device_id=cur_device))
            for cur_pair in cur_pair_list:
                print(padding + padding, counter, "=", cur_pair)
                id_to_profile_pair[counter] = cur_pair
                counter += 1

        return id_to_profile_pair

    @staticmethod
    def get_usb_device(device_id):
        """Get the (first) line in lsusb for device with device_id"""
        cmd = "lsusb | grep " + str(device_id)
        p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = p.communicate()[0].decode("utf-8")
        usb_devices = str(output).splitlines()
        if not usb_devices or len(usb_devices) == 0:
            return []
        else:
            return usb_devices[0]


def parse_args(args):
    if len(args) < 5:
        usage()
        exit()

    # Check chessboard path
    cur_chessboard_path = str(args[1])
    images = glob.glob(cur_chessboard_path)
    if len(images) == 0:
        # TODO[low] - add asterisk manually
        print("NO pictures under chessboard path:", cur_chessboard_path)
        print("Chessboard path should be in the format of:<relative path>/*")
        exit()

    cur_img_path = str(args[2])
    cur_obj_path = str(args[3])
    cur_device_list = set()

    index = 0
    while index < len(args):
        if index > 3:
            cur_device_list.add(str(args[index]))
        index += 1

    return cur_chessboard_path, cur_img_path, cur_obj_path, list(cur_device_list)


def usage():
    print("Usage: python3 undistortion.py <chessboard path> <img point path> "
          "<obj point path> <device1> [<device2> ... <device n>]")
    test_img_points_path = "undistort/profiles/img1"
    test_obj_points_path = "undistort/profiles/obj1"
    print("   <chessboard path>: Path to the folder of chessboard pictures, e.g.\"" +
          default_chessboard_path2 + "\"")
    print("   <img point path> : Path to save the img points without post fix, e.g.\"" +
          test_img_points_path + "\"")
    print("   <obj point path> : Path to save the obj points without post fix, e.g.\"" +
          test_obj_points_path + "\"")
    print("   <device n>       : Device in the format of <idVendor>:<idProduct>, e.g.05a3:9230")


if __name__ == "__main__":
    # Main function is for build profile for a list of devices
    chessboard_path, img_path, obj_path, device_list = parse_args(sys.argv)
    print("chessboard_path", chessboard_path)
    print("img_path", img_path)
    print("obj_path", obj_path)
    print("device_list", device_list)
    Undistortion.chessboard_path_to_profile(chessboard_path, img_path, obj_path, device_list)

    # Example:
    # Undistortion.chessboard_path_to_profile(
    #     default_chessboard_path2, "undistort/profiles/test_img1",
    #     "undistort/profiles/test_obj1", ["05a3:9230", "test"]
    # )
