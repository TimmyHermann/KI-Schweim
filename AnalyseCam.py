import tkinter

import numpy as np
import tensorflow as tf
import cv2
import imutils
import pathlib

from PIL import Image

batch_size = 64
img_height = 108
img_width = 129
(winW, winH) = (129, 108)

model = tf.keras.models.load_model('saved_model/my_model')
Probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

i = 0
e = 0

def pyramid(image, scale=1.5, minSize=(30, 30)):
	# yield the original image
	yield image
	# keep looping over the pyramid
	while True:
		# compute the new dimensions of the image and resize it
		w = int(image.shape[1] / scale)
        #h = int(image.shape[0] / scale)
		image = imutils.resize(image, width=w)
		# if the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break
		# yield the next image in the pyramid
		yield image
def sliding_window(image, stepSize, windowSize):
	# slide a window across the image
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			# yield the current window
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


# define a video capture object
vid = cv2.VideoCapture(0)
while (True):

    # Capture the video frame
    # by frame
    ret, frame = vid.read()
    i = i + 1

    im = Image.fromarray(frame, 'RGB')
    #im = im.resize((img_width,img_height))
    img_array = np.array(im)
    img_array = np.expand_dims(img_array, axis=0)


    if i > 25:
        # loop over the image pyramid
        for resized in pyramid(frame, scale=1.5):
            # loop over the sliding window for each layer of the pyramid
            for (x, y, window) in sliding_window(resized, stepSize=30, windowSize=(winW, winH)):
                # if the window does not meet our desired window size, ignore it
                if window.shape[0] != winH or window.shape[1] != winW:
                    continue
                #status = cv2.imwrite('data/test/img'+str(e)+'.png',window)
                #print("Image written to file-system : ", status)
                wind = Image.fromarray(window, 'RGB')
                window_array = np.array(wind)
                window_array = np.expand_dims(window_array, axis=0)
                print(window_array.shape)
                prediction = int(model.predict(window_array)[0][0])
                #e = e + 1
                if prediction == 1:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    tkinter.messagebox.askokcancel(title=None, message="found one")
                    vid.release()
        i = 0


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
