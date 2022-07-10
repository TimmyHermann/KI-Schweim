# BLUECAM
# JUST FOR FUN, NOT FOR PRODUCTION USE !!!!!
import numpy as np
import cv2 as cv
cap = cv.VideoCapture(0)
while(True):
     # Capture frame-by-frame
     ret, frame = cap.read()
     # Our operations on the frame come here
     color = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
     # Display the resulting frame
     cv.imshow('Hit ESC to close', color)
     if cv.waitKey(1) & 0xFF == 27:
         break
 # When everything done, release the capture
cap.release()
cv.destroyAllWindows()