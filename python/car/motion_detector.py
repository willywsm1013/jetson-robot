import cv2
import numpy as np
import sys

def diffImg(t0, t1, t2):
    d1 = cv2.absdiff(t2, t1)
    d1 = cv2.threshold(d1, 25, 255, cv2.THRESH_BINARY)[1]
    d1 = cv2.dilate(d1, np.ones([5,5], dtype='uint8'), iterations=2)

    d2 = cv2.absdiff(t1, t0)
    d2 = cv2.threshold(d2, 25, 255, cv2.THRESH_BINARY)[1]
    d2 = cv2.dilate(d2, np.ones([5,5], dtype='uint8'), iterations=2)

    return cv2.bitwise_and(d1, d2)

def detect_motion():
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('video.mp4', fourcc ,20.0 ,(640, 480))
    cam = cv2.VideoCapture(0)
    #Read three images first:
    ret, frame = cam.read()
    left = cv2.resize(frame, (640,480))
    t_minus = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    t_minus = cv2.GaussianBlur(t_minus, (21, 21), 0)
    ret, frame = cam.read()
    middle = cv2.resize(frame, (640,480))
    t = cv2.cvtColor(middle, cv2.COLOR_BGR2GRAY)
    t = cv2.GaussianBlur(t, (21, 21), 0)
    ret, frame = cam.read()
    right = cv2.resize(frame, (640,480))
    t_plus = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
    t_plus = cv2.GaussianBlur(t_plus, (21, 21), 0)
    while True:
        diff = diffImg(t_minus, t, t_plus)
        diff = cv2.dilate(diff, np.ones([5,5], dtype='uint8'), iterations=2)
        merge_cnts = FindMovingObj(diff)
        cv2.drawContours(middle, merge_cnts, -1, (0,255,0), 2)
        video.write(middle)
        cv2.imshow('cam', middle)
        # Read next image
        t_minus = t
        left = middle
        t = t_plus
        middle = right
        ret, frame = cam.read()
        right = cv2.resize(frame, (640,480))
        t_plus = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        t_plus = cv2.GaussianBlur(t_plus, (21, 21), 0)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cam.release()
            video.release()
            break
        elif key == ord('s'):
           cv2.imwrite('./motion_example.jpg', diff)

def FindNeighborCnt(cnts):
    num = len(cnts)
    print(num)
    cnts_label = np.arange(num)
    cnts_neighbors = [[] for _ in xrange(num)]
    for i, my_cnt in enumerate(cnts):
        for j in xrange(len(my_cnt)):
            for k, others_l in enumerate(cnts_label):
                my_l = cnts_label[i]
                if my_l == others_l:
                    continue
                else:
                    dist = cv2.pointPolygonTest(cnts[k], tuple(my_cnt[j][0]), True)
                    if dist > -50:
                        min_l = min(my_l, others_l)
                        n_i = cnts_neighbors[i]
                        n_k = cnts_neighbors[k]
                        for neighbor in cnts_neighbors[i]:
                            cnts_label[neighbor] = min_l
                            cnts_neighbors[neighbor] = cnts_neighbors[neighbor] + [k] + n_k
                        for neighbor in cnts_neighbors[k]:
                            cnts_label[neighbor] = min_l
                            cnts_neighbors[neighbor] = cnts_neighbors[neighbor] + [i] + n_i
                        cnts_label[i] = min_l
                        cnts_label[k] = min_l
                        cnts_neighbors[i] = cnts_neighbors[i] + [k] + n_k
                        cnts_neighbors[k] = cnts_neighbors[k] + [i] + n_i
    return cnts_label, cnts_neighbors

def FindMovingObj(img):
    #img = cv2.imread('./motion_example2.jpg')
    #img_gray = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),(640, 480))
    img_gray = cv2.resize(img, (640, 480))
    img_gray = cv2.dilate(img_gray, np.ones([3,3], dtype='uint8'))

    _, cnts, hierarchy = cv2.findContours(img_gray.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [cv2.convexHull(c) for c in cnts]# if cv2.arcLength(c, True) > 5]
    cnts_label, cnts_neighbors = FindNeighborCnt(cnts)
    #print cnts_label
    #print cnts_neighbors

    cnts_check = np.zeros(len(cnts))
    merge_cnts = []
    for i in xrange(len(cnts)):
        if cnts_check[i] == 0:
            cnts_check[i] = 1
            new_cnt = cnts[i]
            for neighbor in cnts_neighbors[i]:
                new_cnt = np.append(new_cnt, cnts[neighbor], axis=0)
                cnts_check[neighbor] = 1
            merge_cnts.append(new_cnt)
    merge_cnts = [cv2.convexHull(c) for c in merge_cnts]
    merge_cnts = [c for c in merge_cnts if cv2.contourArea(c) > 2000]

    '''result = np.full(img_gray.shape, 255, dtype='uint8')
    for i in xrange(len(cnts)):
        cv2.drawContours(result, [cnts[i]], -1, cnts_label[i]*50, 2)'''
    '''result = np.full(img_gray.shape, 255, dtype='uint8')
    cv2.drawContours(result, merge_cnts, -1, 0, 2)
    cv2.imshow('result', result)
    cv2.imshow('orig', img_gray)
    cv2.waitKey(0)'''
    return merge_cnts

if __name__ == "__main__":
    detect_motion()
    cv2.destroyAllWindows()
