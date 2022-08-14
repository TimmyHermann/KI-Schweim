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

model = tf.keras.models.load_model('saved_model/my_model4')
Probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])


print(model.summary())
i = 0
e = 0

# define a video capture object
vid = cv2.VideoCapture(0)
while (True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    i = i + 1

    im = Image.fromarray(frame, 'RGB')
    im = im.resize((img_width,img_height))
    img_array = np.array(im)
    img_array = np.expand_dims(img_array, axis=0)

    #print(img_array.shape)
    prediction = model.predict(img_array,batch_size=None,verbose=0)
    #prediction = model(img_array)
    predi = np.argmax(prediction, axis=1)
    #e = e + 1
    if predi == 1:
        print("Thumbs Up")
    elif predi == 0:
        print("neutral")

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
