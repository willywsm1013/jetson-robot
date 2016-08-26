import cv2
import numpy as np
import time
import threading

class VideoStreaming(threading.Thread):
    def __init__(self, cam, curFrame_cont, curFrame_lock, circles, circles_lock, fin_cont, fin_lock):
        threading.Thread.__init__(self)
        self.vStream = cv2.VideoCapture(cam)
        self.curFrame_cont = curFrame_cont
        self.curFrame_lock = curFrame_lock
        self.circles = circles
        self.circles_lock = circles_lock
        self.fin_cont = fin_cont
        self.fin_lock = fin_lock

    def run(self):
        while True:
            ret, frame = self.vStream.read()
            self.curFrame_lock.acquire()
            self.curFrame_cont[0] = frame
            self.curFrame_lock.release()

            self.circles_lock.acquire()
            circles = self.circles
            self.circles_lock.release()
            if not circles[0] == None:
                circles = np.uint16(np.around(circles))
                for i in circles[0,:]:
                    # draw the outer circle
                    cv2.circle(frame,(i[0],i[1]),i[2],(0,255,0),2)
                    # draw the center of the circle
                    cv2.circle(frame,(i[0],i[1]),2,(0,0,255),3)
            else:
                pass
            cv2.imshow('frame', frame)

            key = cv2.waitKey(20) & 0xFF
            if key == ord('q'):
                self.fin_lock.acquire()
                self.fin_cont[0] = True
                self.fin_lock.release()
                break

class DetectCircle(threading.Thread):
    def __init__(self, curFrame_cont, curFrame_lock, circles, circles_lock, fin_cont, fin_lock):
        threading.Thread.__init__(self)
        self.curFrame_cont = curFrame_cont
        self.curFrame_lock = curFrame_lock
        self.circles = circles
        self.circles_lock = circles_lock
        self.fin_cont = fin_cont
        self.fin_lock = fin_lock

        self.acc_thr = 300

    def run(self):
        while True:
            #img = cv2.imread('./others/door_handle_35.jpg')
            #img = cv2.imread('./positives/door_handle_5.jpg')
            #img = cv2.imread('./test1.jpg')
            #img = cv2.resize(img, (640,480))
            self.curFrame_lock.acquire()
            if not self.curFrame_cont[0] == None:
                frame = self.curFrame_cont[0]
            else:
                self.curFrame_lock.release()
                continue
            self.curFrame_lock.release()

            img_blur = cv2.GaussianBlur(frame, (7,7), 2, 2)

            '''img_yuv = cv2.cvtColor(img_blur, cv2.COLOR_BGR2YCrCb)
            y, u, v = cv2.split(img_yuv)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4,4))
            y = clahe.apply(y)
            img_yuv = cv2.merge([y, u, v])
            img_addcon = cv2.cvtColor(img_yuv, cv2.COLOR_YCrCb2BGR)
            cv2.imshow('addcon', img_addcon)'''

            gray = cv2.cvtColor(img_blur,cv2.COLOR_BGR2GRAY)
            img_edge = cv2.Canny(gray, 50, 5)
            img_edge = cv2.dilate(img_edge, np.ones([5,5], dtype='uint8'), iterations=2)
            #img_edge = cv2.erode(img_edge, np.ones([9,9], dtype='uint8'), iterations=2)
            #img_edge = cv2.morphologyEx(img_edge, cv2.MORPH_CLOSE, np.ones([7,7], dtype='uint8'))

            circles = cv2.HoughCircles(img_edge,cv2.HOUGH_GRADIENT,1,100,param1=2,param2=self.acc_thr,minRadius=0,maxRadius=0)
            try:
                if circles.shape[1] > 5:
                    self.acc_thr += 10
                self.circles_lock.acquire()
                self.circles[0] = circles[0]
                self.circles_lock.release()
            except:
                self.circles_lock.acquire()
                self.circles[0] = None
                self.circles_lock.release()
                self.acc_thr -= 5
            print self.acc_thr
            self.fin_lock.acquire()
            fin = self.fin_cont[0]
            self.fin_lock.release()
            if fin == True:
                break

if __name__ == "__main__":
    curFrame_cont = [None]
    curFrame_lock = threading.Lock()
    circles = [None]
    circles_lock = threading.Lock()
    fin_cont = [False]
    fin_lock = threading.Lock()

    video_thread = VideoStreaming(0, curFrame_cont, curFrame_lock, circles, circles_lock, fin_cont, fin_lock)
    #det_cir_thread = DetectCircle(curFrame_cont, curFrame_lock, circles, circles_lock, fin_cont, fin_lock)

    video_thread.start()
    #det_cir_thread.start()

    video_thread.join()
    #det_cir_thread.join()

    cv2.destroyAllWindows()
