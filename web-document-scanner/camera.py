import cv2
from document import Scanner

class VideoCamera(object):
    def __init__(self):
        # Open a camera
        self.cap = cv2.VideoCapture(0)
      
        # Initialize video recording environment
        self.is_record = False
        self.out = None
        self.transformed_frame = None

        self.scanner = Scanner()
        self.cached_frame = None
    
    def __del__(self):
        self.cap.release()

    def get_video_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # frame, _ = self.scanner.detect_edge(frame)
            try:
                output = self.scanner.hough(frame)
                if output is None:
                    return self.cached_frame
                # print (output)
                self.cached_frame = output
                ret, jpeg = cv2.imencode('.jpg', output)
                return jpeg.tobytes()
            except:
                return self.cached_frame
        else:
            return self.cached_frame

    def capture_frame(self):
        ret, frame = self.cap.read()
        if ret:
            _, frame = self.scanner.detect_edge(frame, True)
            ret, jpeg = cv2.imencode('.jpg', frame)
            self.transformed_frame = jpeg.tobytes()
        else:
            return None

    def get_cached_frame(self):
        return self.cached_frame

    def get_image_frame(self):
        return self.transformed_frame

