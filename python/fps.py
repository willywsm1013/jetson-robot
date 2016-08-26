from __future__ import print_function
from imutils.video import VideoStream
from imutils.video import FPS
import argparse
import imutils
import cv2
import dlib

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--display", type=int, default=-1,
    help="Whether or not frames should be displayed")
args = vars(ap.parse_args())

print("[INFO] sampling THREADED frames from webcam...")
vs = VideoStream(src=0).start()
fps = FPS().start()
cv2.namedWindow("MAIN")

# loop over some frames...this time using the threaded stream
i = 0
while True:
    i  = (i+1) % 100
    frame = vs.read()
    if args["display"] > 0 and i == 0:
        print("go")
        frame = imutils.resize(frame, width=200)
        cv2.imshow("MAIN", frame)
        cv2.waitKey(30)
    fps.update()
"""
    dlib.find_candidate_object_locations(frame, rects, min_size=6000)
    for k, d in enumerate(rects):
        cv2.rectangle(frame, (d.left(), d.top()), (d.right(), d.bottom()), (0, 255, 0), 1)
"""

# stop the timer and display FPS information
vs.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
fps.stop()
