import cv2
from threading import Thread
import time
import numpy as np
class find_point:
    def __init__(self):
        self.img=None
        self.kp=None
        self.des=None
    def start(self):
        th=Thread(target=self.find,args=())
        th.start()
        time.sleep(0)
        th.join()
    def find(self):
        while(True):
            detector=cv2.SURF(400,6,6)
            matcher=cv2.BFMatcher(cv2.NORM_L2)
            self.kp, self.des=detector.detectAndCompute(self.img, None)
            self.key_size()
            break

    def set_img(self,im):
        self.img=cv2.imread(im)
    def set_cv_img(self,im):
        self.img=im
    def get_point(self):
        return self.kp, self.des
    def key_size(self):
        kp_=[]
        des_=[]
        kp_size=np.float32([self.kp[i].size for i in range(len(self.kp))])

        for i in range(len(self.kp)):
            if self.kp[i].size>20:
                kp_.append(self.kp[i])
                des_.append(self.des[i])
        self.kp=None
        self.kp=kp_
        self.des=None
        self.des=np.float32(des_)
