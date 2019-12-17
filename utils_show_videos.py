import cv2
import main
import math
import os
import shutil

if __name__== "__main__":
    dir = 'data'
    labels = {}
    videos = {}
    for file in os.listdir(dir):
        items = file.split(".")
        filename = items[0]
        extension = items[1]
        if extension == "mov" or extension == "mp4":
            videos[filename] = dir + "/" + file
    for filename in videos:
        
        video_path = videos[filename]
        print (video_path)
        cam_device = cv2.VideoCapture(video_path)
        
        while True:
            try:
                ret, im = cam_device.read()
                # cv2.imwrite("output/frame_" + str(self.frame_num)+"_original.jpg", orig)
                if not ret:
                    print("End of Input Stream")
                    break
                cv2.imshow('input image', im)
                cv2.waitKey(0)
            except Exception as e:
                print(e)
                break