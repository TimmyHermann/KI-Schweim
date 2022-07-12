import numpy as np
import tensorflow as tf
import cv2
import pathlib

# data_dir =('data')
# data_dir = pathlib.Path(data_dir)
#
# image_count = len(list(data_dir.glob('*/*.jpg')))
#
from PIL import Image

batch_size = 64
img_height = 108
img_width = 129
#
# train_ds = tf.keras.utils.image_dataset_from_directory(
#   data_dir,
#   validation_split=0.2,
#   subset="training",
#   seed=123,
#   image_size=(img_height, img_width),
#   batch_size=batch_size)
#
# val_ds = tf.keras.utils.image_dataset_from_directory(
#   data_dir,
#   validation_split=0.2,
#   subset="validation",
#   seed=123,
#   image_size=(img_height, img_width),
#   batch_size=batch_size)
#
model = tf.keras.models.load_model('saved_model/my_model')
Probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])

#
# #model evaluation
# print("\nTraining Set: ")
# model.evaluate(train_ds, verbose=2)
# print("\nTest Set: ")
# model.evaluate(val_ds, verbose=2)
#
i = 0
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


    if i > 10:
        prediction = int(model.predict(img_array)[0][0])
        print(prediction)
        if prediction == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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

# def live_input_fn():
#     ret, frame = cv2.VideoCapture.read()
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#
#     frame = tf.image.convert_image_dtype(frame, dtype=tf.float32)
#     frame = tf.image.per_image_standardization(frame)
#
#     frame = tf.reshape(frame, [1, 120, 160, 3])
#     dataset = tf.contrib.data.Dataset.from_tensor_slices(frame)
#
#     iterator = dataset.make_one_shot_iterator()
#
#     features = iterator.get_next()
#     return features, None
#
# Probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
#
# while True:
#     predictions = Probability_model.predict(
#         input_fn=live_input_fn
#     )
#     for p in predictions:
#         print([p["classes"]])
#         print(p["probabilities"])