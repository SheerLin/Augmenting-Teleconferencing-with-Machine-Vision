import cv2
import glob
import os
import time
import numpy as np

profiles_folder = "undistort/profiles"
npy_file_postfix = ".npy"

default_chessboard_path = "undistort/data/chessboard/original/*"
default_chessboard_path2 = "undistort/data/chessboard/original4/*"

# Set default_img_points_path and default_obj_points_path
default_img_points_path = profiles_folder + "/img1" + npy_file_postfix
default_obj_points_path = profiles_folder + "/obj1" + npy_file_postfix

# TODO - needed for selecting profile
para_2_profile_file_path = ""


# original_chessboard_path = 'data/chessboard/original/*.jpg'
# to_calibrate_path = 'data/distorted/'
# imgpoints_profile_path = 'profiles/profile1_imgpoints.txt'
# objpoints_profile_path = 'profiles/profile1_objpoints.txt'


#  1. If has chessboards images folder path, use this path to get point.
#  2. if has profile path, use this profile
#  3. If no profile path and no has chessboards images folder path, select from the existing profiles
#  4. undistort the image


class Undistortion:
    def __init__(self, chessboard_folder_path=None, img_points_path=None, obj_points_path=None):
        self.img_points_path = img_points_path
        self.obj_points_path = obj_points_path
        self.chessboard_folder_path = chessboard_folder_path

        # TO be initialized:
        # Arrays to store object points and image points from all the images.
        self.obj_points = []  # 3d point in real world space
        self.img_points = []  # 2d points in image plane.
        self.para_to_profile = dict()  # Profile for selecting profile

        # debug
        self.show_image = False
        self.save_image = False
        self.both_way = False
        self.crop = True
        # self.imshow_size = cv2.WINDOW_FULLSCREEN
        self.imshow_size = cv2.WINDOW_NORMAL

        self.initialize()

    def __call__(self, img):
        return self.calibrate_image(img)

    def initialize(self):
        """Initialize the needed info"""
        print("self.chessboard_folder_path", self.chessboard_folder_path)
        print("self.img_points_path", self.img_points_path)
        print("self.obj_points_path", self.obj_points_path)
        # 1. Init self.obj_points and self. img_points
        # if self.chessboard_folder_path and os.path.exists(self.chessboard_folder_path):
        if self.chessboard_folder_path:
            print("Use input chessboard folder path:", self.chessboard_folder_path)
            self.img_points, self.obj_points = self.__init_img_obj_points_from_chessboards(self.chessboard_folder_path)
        else:
            if self.img_points_path and os.path.isfile(self.img_points_path) \
                    and self.obj_points_path and os.path.exists(self.obj_points_path):
                pass
            else:
                self.img_points_path, self.obj_points_path = self.__select_profile()
                if not self.img_points_path or not self.obj_points_path:
                    print("Failed to initialize profile path.")
                    return False

            print("Use img_points_path:", self.img_points_path)
            print("Use obj_points_path:", self.obj_points_path)

            self.img_points, self.obj_points = Undistortion.deserialize(self.img_points_path, self.obj_points_path)
        # 2. Init self.para_to_profile
        self.__init_para_to_profile()

    def __init_img_obj_points_from_chessboards(self, chessboard_path):
        """Get image points and object points from the pictures in the chessboard_path"""
        start_time = time.time()
        print("in __init_img_obj_points_from_chessboards")

        # termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((6 * 7, 3), np.float32)
        objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

        images = glob.glob(chessboard_path)
        valid_pics = 0
        obj_points = []
        img_points = []

        if len(images) == 0:
            print("NO pictures under chessboard path:", chessboard_path)
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
                if self.show_image:
                    img = cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
                    cv2.imshow('img', img)
                    cv2.waitKey(100)

                valid_pics += 1
                print("Initialized from ", valid_pics, "pictures:", file_name)
            else:
                print("Failed to findChessboardCorners. Skip current image:", file_name)

        elapsed_time = time.time() - start_time
        cv2.destroyAllWindows()
        print("Collect 3d point from ", valid_pics, " pictures:", elapsed_time)
        print("img_points", type(img_points), "len=", len(img_points))
        print("obj_points", type(obj_points), "len=", len(obj_points))

        return img_points, obj_points

    def __init_para_to_profile(self):
        if para_2_profile_file_path and os.path.isfile(para_2_profile_file_path):
            # TODO - read from para_2_profile_file_path to construct self.para_to_profile
            print(self.para_to_profile)
            pass
        else:
            return False

    def __select_profile(self):
        """1. Find out camera parameter
        2. Find the corresponding profile according to parameters"""

        if len(self.para_to_profile.keys()) > 0:
            # 1. Find out camera parameter
            para = Undistortion.get_cam_para()
            # TODO - 2.Find the corresponding profile according to parameters"
            pass

        elif default_img_points_path and os.path.isfile(default_img_points_path and default_obj_points_path) \
                and os.path.exists(default_obj_points_path):
            return default_img_points_path, default_obj_points_path

        else:
            return False

    def chessboard_path_to_profile(self, chessboard_path, img_points_path, obj_points_path):
        """1. Save the points from chessboard path pictures to profile_path
        2. Take the parameter of current cam and consider these pictures are captured by this cam
        3. self.define_para_to_profile: Update para_2_profile_file_path file and self.para_to_profile"""
        img_points, obj_points = self.__init_img_obj_points_from_chessboards(chessboard_path)
        Undistortion.serialize(img_points, obj_points, img_points_path, obj_points_path)

        para = Undistortion.get_cam_para()
        self.define_para_to_profile(para, img_points_path, obj_points_path)

    @staticmethod
    def get_cam_para():
        # TODO - Get the parameter
        para = {}
        return para

    def define_para_to_profile(self, para, img_points_path, obj_points_path):
        """1. Write to para_2_profile_file_path file
        2. Update in memory self.para_to_profile"""
        # TODO define_para_to_profile
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
    def deserialize(img_path, obj_path):
        if not img_path or not os.path.exists(img_path):
            print("img_path(", img_path + "):File not exist")
            return

        if not obj_path or not os.path.exists(obj_path):
            print("obj_path(", obj_path + "):File not exist")
            return

        img_points = np.load(img_path)
        obj_points = np.load(obj_path)

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
        # start_time = time.time()
        # print("Start calibrate_image")
        if self.obj_points is None or self.img_points is None:
            print("not self.obj_points or not self.img_points")

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # print("gray", gray)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.obj_points, self.img_points,
                                                           gray.shape[::-1], None, None)
        # print("mtx", mtx)  # no change
        # print("dist", dist)  # no change

        h, w = image.shape[:2]
        new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
        # print("new_camera_mtx", new_camera_mtx)  # no change

        # cv2.namedWindow("before undistortion", self.imshow_size)
        # cv2.imshow("before undistortion", image)

        dst = cv2.undistort(image, mtx, dist, None, new_camera_mtx)
        # mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, new_camera_mtx, (w, h), 5)
        # dst = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)

        # cv2.namedWindow("after undistortion", self.imshow_size)
        # cv2.imshow("after undistortion", dst)

        if self.crop:
            # crop the image
            x, y, w, h = roi
            dst = dst[y:y + h, x:x + w]
            # cv2.namedWindow("crop", self.imshow_size)
            # cv2.imshow("crop", dst)

        # cv2.waitKey(10000)
        # print("haha")

        # elapsed_time = time.time() - start_time
        # print("Undistort in :", elapsed_time)

        # cv2.destroyAllWindows()
        return dst

    def calibrate_images(self, to_calibrate_path):
        """Calibrate all pictures under to_calibrate_path"""
        start_time = time.time()
        # TODO - what is gray?
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(self.obj_points, self.img_points,
                                                           gray.shape[::-1], None, None)

        for f in glob.glob(to_calibrate_path + 'calibrated/*'):
            os.remove(f)

        distorted_images = glob.glob(to_calibrate_path + 'original/*.jpg')
        index = 1
        print("Start calibrate images in:", to_calibrate_path)

        for file_name in distorted_images:
            print(file_name)
            img = cv2.imread(file_name)

            h, w = img.shape[:2]
            new_camera_mtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

            # Show original
            if self.show_image:
                cv2.imshow('img', img)
                cv2.waitKey(100)

            if self.save_image:
                cv2.imwrite(to_calibrate_path + 'calibrated/' + str(index) + '_0original.jpg', img)

            # method1: undistort using undistort()
            dst = cv2.undistort(img, mtx, dist, None, new_camera_mtx)

            # Show undistorted
            if self.show_image:
                cv2.imshow('img', dst)
                cv2.waitKey(100)

            if self.save_image:
                cv2.imwrite(to_calibrate_path + 'calibrated/' + str(index) + '_1undistort.jpg', dst)

            if self.crop:
                # crop the image
                x, y, w, h = roi
                dst = dst[y:y + h, x:x + w]

                # Show undistorted and cropped
                if self.show_image:
                    cv2.imshow('img', dst)
                    cv2.waitKey(300)

                if self.save_image:
                    cv2.imwrite(to_calibrate_path + 'calibrated/' + str(index) + '_1undistort_cropped.jpg', dst)

            if self.both_way:
                # method2: undistort using remap
                mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, new_camera_mtx, (w, h), 5)
                dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

                # Show original
                if self.show_image:
                    cv2.imshow('img', img)
                    cv2.waitKey(100)

                    # Show undistorted
                    cv2.imshow('img', dst)
                    cv2.waitKey(100)

                if self.save_image:
                    cv2.imwrite(to_calibrate_path + 'calibrated/' + str(index) + '_2remap.jpg', dst)

                if self.crop:
                    # crop the image
                    x, y, w, h = roi
                    dst = dst[y:y + h, x:x + w]

                    # Show undistorted and cropped
                    if self.show_image:
                        cv2.imshow('img', dst)
                        cv2.waitKey(600)

                    if self.save_image:
                        cv2.imwrite(to_calibrate_path + 'calibrated/' + str(index) + '_2remap_cropped.jpg', dst)

            index += 1

        elapsed_time = time.time() - start_time
        print("Undistort", len(distorted_images), " pictures:", elapsed_time)

        cv2.destroyAllWindows()
