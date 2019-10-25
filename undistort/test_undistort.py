import undistortion
import cv2

original_chessboard_path = 'data/chessboard/original/*'
profile_path = "profiles/test1.txt"
img_path = "data/distorted/original/2wb1.jpg"

undistort_instance = undistortion.Undistortion(profile_path=None,
                                               chessboard_folder_path=original_chessboard_path)

# undistort_instance.set_profile_path(profile_path)
# undistort_instance.initialize()

# img = cv2.imread(img_path)
# cv2.imshow('img', img)
# cv2.waitKey(400)

# new_img = undistort_instance.calibrate_image(img)

# cv2.imshow('img', new_img)
# cv2.waitKey(500)

original_img_points, original_obj_points = undistortion.Undistortion.serialize(undistort_instance.img_points,
                                                                               undistort_instance.obj_points,
                                                                               profile_path)

new_img_points, new_obj_points = undistortion.Undistortion.deserialize(profile_path)

# undistortion.Undistortion.check_np_array_equals(original_img_points, new_img_points)
# undistortion.Undistortion.check_np_array_equals(original_obj_points, new_obj_points)
