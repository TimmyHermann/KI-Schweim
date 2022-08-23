import tkinter
import numpy as np
import tensorflow as tf
import cv2
import imutils
from PIL import Image
from matplotlib import pyplot as plt

batch_size = 64
img_height = 108
img_width = 129
(winW, winH) = (129, 108)
predi = np.array([0, 0])

model = tf.keras.models.load_model('saved_model/my_model5')
# Probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])


l = 0
blocked = False

# define a video capture object
vid = cv2.VideoCapture(0)
while (True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()


    im = Image.fromarray(frame, 'RGB')
    im = im.resize((img_width, img_height))
    img_array = np.array(im)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, batch_size=None, verbose=0)
    print(prediction)

    if np.argmax(prediction, axis=1) == 0:
        blocked = False

    if l == 10:
        if not blocked:
            predi = np.append(predi, np.argmax(prediction, axis=1))
            if sum(predi[-3:]) == 3:
                print("Thumbs Up")
                blocked = True
            predi = predi[-6:]
        l = 0
    l = l + 1

    cv2.imshow("Capturing", frame)
    # the 'q' button is set as the
    # quitting button you may use any
    # desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# After the loop release the cap object
vid.release()
# Destroy all the windows
cv2.destroyAllWindows()
