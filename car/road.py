import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    adapt = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    adapt = cv2.adaptiveThreshold(adapt, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY,
                                  65, 10)
    # Display the resulting frame
    mask = (adapt == 255)
    print(frame)
    # frame = frame[mask]
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
