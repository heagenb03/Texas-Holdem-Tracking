import cv2 as cv
from threading import Thread

class VideoStream:
    def __init__(self):
        self.cap = cv.VideoCapture(0)
        if not self.cap.isOpened():
            raise Exception("Can't open camera")
        
        (self.grabbed, self.frame) = self.cap.read()
        
        self.stopped = False
    
    def start(self):
        Thread(target=self.update, args=()).start()
        return self
    
    def update(self):
        while True:
            if self.stopped:
                self.cap.release()
                return
            
            (self.grabbed, self.frame) = self.cap.read()
    
    def read(self):
        return self.frame
    
    def stop(self):
        self.stopped = True