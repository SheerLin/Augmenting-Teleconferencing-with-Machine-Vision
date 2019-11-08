import numpy as np
import cv2
import glob
import os
import time
import json
import numpy as np
import codecs
import undistortion


def undistort_square(src):
    return src


original_chessboard_path = 'data/chessboard/original/*.jpg'
to_calibrate_path = 'data/distorted/'
imgpoints_profile_path = 'profiles/profile1_imgpoints.txt'
imgpoints_profile_path2 = 'profiles/profile1_imgpoints2.txt'
objpoints_profile_path = 'profiles/profile1_objpoints.txt'

show_image = False
save_image = False
both_way = False
crop = False
start_time = time.time()

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6 * 7, 3), np.float32)
objp[:, :2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.

images = glob.glob(original_chessboard_path)
valid_pics = 0

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (7, 6), None)

    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)

        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (7, 6), corners2, ret)
        if show_image:
            cv2.imshow('img', img)
            cv2.waitKey(100)

        valid_pics += 1
    else:
        print("No return value")

elapsed_time = time.time() - start_time
cv2.destroyAllWindows()
print("Collect 3d point from ", valid_pics, " valid pictures after scanning all ", len(images), " pictures:",
      elapsed_time)


# index = 0
# dictionary = dict()
# print("imgpoints", type(imgpoints), "len=", len(imgpoints))
# print("objpoints", type(objpoints), "len=", len(objpoints))

# for imgpoint in imgpoints:
#     # print("imgpoint", type(imgpoint), imgpoint)
#     # json.dumps(imgpoint)
#     imgpoint_list = imgpoint.tolist()
#     dictionary[index] = imgpoint_list
#
#     index += 1
#
# dictionary_dump = json.dumps(dictionary)
# with open(imgpoints_profile_path, 'w+') as filename:
#     filename.writelines(dictionary_dump)
#
# json.dump(dictionary_dump, codecs.open(imgpoints_profile_path2, 'w', encoding='utf-8'), separators=(',', ':'),
#           sort_keys=True, indent=4)
#
# obj_text = codecs.open(imgpoints_profile_path, 'r', encoding='utf-8').read()
# dictionary_load = json.loads(obj_text)
# print("dictionary_load", len(dictionary_load))
#
# index = 0
# for de_imgpoint in dictionary_load.values():
#     de_imgpoint_np = np.array(de_imgpoint)
#     imgpoint_np = imgpoints[index]
#     index += 1
#     # print(np.equal(de_imgpoint_np, imgpoint_np))
#
#
# exit()

start_time = time.time()
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

for f in glob.glob(to_calibrate_path + 'calibrated/*'):
    os.remove(f)

distorted_images = glob.glob(to_calibrate_path + 'original/*.jpg')
index = 1

for fname in distorted_images:
    img = cv2.imread(fname)

    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # Show original
    if show_image:
        cv2.imshow('img', img)
        cv2.waitKey(100)

    if save_image:
        cv2.imwrite(to_calibrate_path + 'calibrated/' + str(index) + '_0original.jpg', img)

    # method1: undistort using undistort()
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # Show undistorted
    if show_image:
        cv2.imshow('img', dst)
        cv2.waitKey(100)

    if save_image:
        cv2.imwrite(to_calibrate_path + 'calibrated/' + str(index) + '_1undistort.jpg', dst)

    if crop:
        # crop the image
        x, y, w, h = roi
        dst = dst[y:y + h, x:x + w]

        # Show undistorted and cropped
        if show_image:
            cv2.imshow('img', dst)
            cv2.waitKey(300)

        if save_image:
            cv2.imwrite(to_calibrate_path + 'calibrated/' + str(index) + '_1undistort_cropped.jpg', dst)

    if both_way:
        # method2: undistort using remap
        mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
        dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

        # Show original
        if show_image:
            cv2.imshow('img', img)
            cv2.waitKey(100)

            # Show undistorted
            cv2.imshow('img', dst)
            cv2.waitKey(100)

        if save_image:
            cv2.imwrite(to_calibrate_path + 'calibrated/' + str(index) + '_2remap.jpg', dst)

        if crop:
            # crop the image
            x, y, w, h = roi
            dst = dst[y:y + h, x:x + w]

            # Show undistorted and cropped
            if show_image:
                cv2.imshow('img', dst)
                cv2.waitKey(600)

            if save_image:
                cv2.imwrite(to_calibrate_path + 'calibrated/' + str(index) + '_2remap_cropped.jpg', dst)

    index += 1

elapsed_time = time.time() - start_time
print("Undistort", len(distorted_images), " pictures:", elapsed_time)

# mean_error = 0
# for i in xrange(len(objpoints)):
#     imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
#     error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
#     tot_error += error
#
# print("total error: " + str(mean_error / len(objpoints)))

cv2.destroyAllWindows()




def test_undistort():
    # original_chessboard_path = 'undistort/data/chessboard/original/*'
    profile_path = "undistort/profiles/test2.txt"
    img_path = "undistort/data/distorted/original3/q7_1.png"

    img_points_path_no_post_fix = "undistort/profiles/img1"
    img_points_path_with_post_fix = "undistort/profiles/img1" + undistortion.npy_file_postfix
    obj_points_path_no_post_fix = "undistort/profiles/obj1"
    obj_points_path_with_post_fix = "undistort/profiles/obj1" + undistortion.npy_file_postfix

    # Test 1. Use chessboard folder
    # undistort_instance = undistortion.Undistortion(chessboard_folder_path=undistortion.default_chessboard_path2)
    #
    # original_img_points, original_obj_points = undistortion.Undistortion.serialize(undistort_instance.img_points,
    #                                                                                undistort_instance.obj_points,
    #                                                                                img_points_path_no_post_fix,
    #                                                                                obj_points_path_no_post_fix)
    # new_img_points, new_obj_points = \
    #     undistortion.Undistortion.deserialize(img_points_path_with_post_fix, obj_points_path_with_post_fix)
    #
    # undistortion.Undistortion.check_np_array_equals(original_img_points, new_img_points)
    # undistortion.Undistortion.check_np_array_equals(original_obj_points, new_obj_points)

    # Test 2. Use profile
    undistort_instance = undistortion.Undistortion(chessboard_folder_path=None,
                                                   img_points_path=img_points_path_with_post_fix,
                                                   obj_points_path=obj_points_path_with_post_fix)

    #################################
    # Test single image calibration
    #################################
    img = cv2.imread(img_path)
    new_img = undistort_instance(img)

    # cv2.resizeWindow('image', 600,600)
    cv2.waitKey(10000)

    cv2.destroyAllWindows()