import cv2
from PIL import Image
import numpy as np

e = 0
i = 0

vid = cv2.VideoCapture(0)
while True:

    ret, frame = vid.read()

    status = cv2.imwrite("data/red/frame" + str(i) + ".jpg", frame)
    print(status)

    cv2.imshow("Capturing", frame)
    i = +1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vid.release()