import numpy as np
import cv2
from document import Scanner

cap = cv2.VideoCapture(2)
scanner = Scanner()

while(cap.isOpened()):
    
    ret, frame = cap.read()
    video_frame = None
    image_frame = None

    if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if ret:
        if cv2.waitKey(1) & 0xFF == ord('p'):
            video_frame, image_frame = scanner.detect_edge(frame, True)
        else:
            video_frame, _ = scanner.detect_edge(frame)

        if video_frame is not None:
            cv2.imshow("Quadrilateral", video_frame)

        if image_frame is not None:
            cv2.imshow("Transfrom", image_frame)

# Release everything if job is finished
cap.release()
cv2.destroyAllWindows()