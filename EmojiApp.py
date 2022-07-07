# import the opencv library
import cv2
  
  
# define a video capture object
vid = cv2.VideoCapture(0)
  
while(True):
      
    # Capture the video frame
    # by frame
    ret, frame = vid.read()
  
    # Display the resulting frame
    cv2.imshow('frame', frame)
      
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()


# BLUECAM
# import numpy as np
# import cv2 as cv
# cap = cv.VideoCapture(0)
# while(True):
#     # Capture frame-by-frame
#     ret, frame = cap.read()
#     # Our operations on the frame come here
#     color = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
#     # Display the resulting frame
#     cv.imshow('Hit ESC to close', color)
#     if cv.waitKey(1) & 0xFF == 27:
#         break
# # When everything done, release the capture
# cap.release()
# cv.destroyAllWindows()