import cv2
import main
import math
import os
import shutil

if __name__== "__main__":
    dir = 'data'
    videos = {} # filename without extension-> path to file
    labels = {} # filename without extension -> path to file
    diffs = []
    for file in os.listdir(dir):
        items = file.split(".")
        filename = items[0]
        extension = items[1]
        if extension == "txt": # label file
            labels[filename] = dir + "/" + file
        elif extension == "log":
            continue
        else:
            videos[filename] = dir + "/" + file
    for filename in videos:
        video_path = videos[filename]
        label_path = labels[filename]
        log_path = dir + "/" + filename + ".log"
        print (video_path)
        print (label_path)
        print (log_path)
        if os.path.exists("extract_points.log"):
            os.remove('extract_points.log')
        cam_device = cv2.VideoCapture(video_path)
        main.process_video(cam_device, None)
        shutil.copyfile('extract_points.log', log_path)
        width = cam_device.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
        height = cam_device.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float
        del (cam_device)
        target_points = [[0,0],[0,0],[0,0],[0,0]]
        with open(label_path, "r") as benchmark_file:
            line = benchmark_file.readline()
            points = line.split(";")[:-1]
            for idx, point in enumerate(points):
                p = point.split(",")
                target_points[idx][0] = float(p[0])
                target_points[idx][1] = float(p[1])
        print(target_points)


        print("log_path", log_path)
        with open(log_path, "r") as log_file:
            cnt = 0
            diff_total = 0
            for line in log_file:
                cnt+=1
                # print("Line {}: {}".format(cnt, line))
                points = line.split(";")[:-1]
                diff = 0
                for idx, point in enumerate(points):
                    p = point.split(",")
                    # print("point", point, "target",target_points[idx])
                    diff += math.sqrt((float(p[0]) - target_points[idx][0]) ** 2 + (float(p[1]) - target_points[idx][1])**2)

                diff_total += diff / 4
            print ("cnt", cnt   )
            diff_percentage = round(100 * diff_total / cnt / math.sqrt(width ** 2 + height ** 2), 2)
            print ("cnt", cnt, "width", width, "height", height, "diff", round(diff_total/cnt,2),
                   "(", diff_percentage , "%)")
            diffs.append(diff_percentage)
    print ('avg diff percentage', sum(diffs) / float(len(diffs)), "%")

