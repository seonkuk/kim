import cv2
from threading import Thread

  
class Webcam:
  
    def __init__(self):
        self.th=None
        self.video_capture = cv2.VideoCapture(0)
        self.current_frame = self.video_capture.read()[1]
          
    # create thread for capturing images
    def start(self):
        self.th=Thread(target=self._update_frame, args=())
        self.th.start()
  
    def _update_frame(self):
        while(True):
            self.current_frame = self.video_capture.read()[1]
                  
    # get the current frame
    def get_current_frame(self):
        return self.current_frame
    def get_frame_shape(self):
        return self.current_frame.shape
    def finish(self):
        self.video_capture.release()
        self.th.join()
