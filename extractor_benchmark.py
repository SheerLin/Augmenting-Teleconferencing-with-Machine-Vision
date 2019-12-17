import os
import shutil
import time

import cv2
from polygon_intersection.polygon_intersection_area import getScore

import extractor
import main

benchmark_logger = None
dir = 'data/benchmark'

def log(str):
    print (str)
    benchmark_logger.write(str + '\n')

if __name__== "__main__":
    start_time = time.time()
    videos = {} # filename without extension-> path to file
    labels = {} # filename without extension -> path to file
    f1_scores = []
    precisions = []
    recalls = []
    
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
    benchmark_logger = open(dir + '/benchmark_result.log', "w")
    
    for filename in videos:
        video_path = videos[filename]
        log('---------------------- ' + video_path + ' ---------------------')
        label_path = labels[filename]
        log_path = dir + "/" + filename + ".log"
        log(video_path)
        log(label_path)
        log(log_path)
        if os.path.exists("extract_points.log"):
            os.remove('extract_points.log')

        cam_device = cv2.VideoCapture(video_path)
        width = int(cam_device.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cam_device.get(cv2.CAP_PROP_FRAME_HEIGHT))

        cap_device = None
        img_path = None
        obj_path = None
        main.process_video(
            cam_device, cap_device,
            width, height,
            img_path, obj_path, 
            {
                'vcam': False,
                'undistorter': False,
                'beautifier': False,
                'benchmark': True,
                'debug': False,
            }
        )
        shutil.copyfile('extract_points.log', log_path)
        
        del (cam_device)
        output_points = [[0,0],[0,0],[0,0],[0,0]]
        target_points = [[0,0],[0,0],[0,0],[0,0]]
        with open(label_path, "r") as benchmark_file:
            line = benchmark_file.readline()
            points = line.split(";")[:-1]
            for idx, point in enumerate(points):
                p = point.split(",")
                target_points[idx][0] = float(p[0])
                target_points[idx][1] = float(p[1])

        with open(log_path, "r") as log_file:
            cnt = 0
            f1_score_total = 0
            precision_total = 0
            recall_total = 0
            for line in log_file:
                cnt+=1
                points = line.split(";")[:-1]
                for idx, point in enumerate(points):
                    p = point.split(",")
                    output_points[idx][0] = float(p[0]) 
                    output_points[idx][1] = float(p[1])
                precision, recall, f1 = getScore(output_points, target_points, width, height)
                f1_score_total += f1
                precision_total += precision
                recall_total += recall
            f1_score_avg = f1_score_total / cnt
            precision_avg = precision_total / cnt
            recall_avg = recall_total / cnt
            f1_scores.append(f1_score_avg)
            precisions.append(precision_avg)
            recalls.append(recall_avg)
            log('sample output points: ' + str(output_points))
            log('sample target points: ' + str(target_points))
            log('avg precision for this video: ' + str(precision_avg))
            log('avg recall for this video: ' + str(recall_avg))
            log('avg f1_score for this video: ' + str(f1_score_avg))
            log('---------------------- ' + video_path + ' ---------------------\n\n')
    
    elapsed_time = time.time() - start_time
    log('\n\n---------------------- benchmark result ---------------------')
    log('processed '+ str(len(videos)) +' videos')
    log('elapsed time: ' + str(int(elapsed_time)) + ' seconds')
    log('avg precisionfor all videos: ' + str(sum(precisions) / float(len(precisions))))
    log('avg recall for all videos: ' + str(sum(recalls) / float(len(recalls))))
    log('avg f1_score for all videos: ' + str(sum(f1_scores) / float(len(f1_scores))))
    log('benchmark results saved to: ' + dir + '/benchmark_result.log')
    benchmark_logger.close()
