# imports
import tensorflow as tf
import matplotlib.pyplot as plt
from datetime import datetime
import random
import numpy as np


# load MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


# plot 10 example images
num = 10
images = x_train[:num]
labels = y_train[:num]

num_row = 2
num_col = 5

fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
for i in range(num):
    ax = axes[i//num_col, i%num_col]
    ax.imshow(images[i], cmap='gray_r')
    ax.set_title('Label: {}'.format(labels[i]))
plt.tight_layout()
plt.show()


# build first model ("Vanilla ANN")
neurons = 16
dropout_rate = 0.2
batch_size = 64
epochs = 1
model_version = 1

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(neurons, activation='relu'),
  tf.keras.layers.Dropout(dropout_rate),
  tf.keras.layers.Dense(10)
])
# get logits
predictions = model(x_train[:1]).numpy()
# interpret as probabilities
tf.nn.softmax(predictions).numpy()
# calculate loss
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
# compile the model
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

print(model.summary())

# train the model
model.fit(
    x_train,
    y_train, 
    batch_size=batch_size,
    epochs=epochs)

# model evaluation
print("\nTraining Set: ")
model.evaluate(x_train,  y_train, verbose=2)
print("\nTest Set: ")
model.evaluate(x_test,  y_test, verbose=2)



x_train_ = np.expand_dims(x_train, -1)
x_test_ = np.expand_dims(x_test, -1)
y_train_ = tf.keras.utils.to_categorical(y_train, 10)
y_test_ = tf.keras.utils.to_categorical(y_test, 10)

dropout_rate = 0.5
batch_size = 128
epochs = 2
model_version = 2

model2 = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(28, 28, 1)),
        tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(10, activation="softmax"),
    ]
)

model2.summary()
# compile model
model2.compile(optimizer="adam",
               loss="categorical_crossentropy",  
               metrics=["accuracy"])

model2.fit(x_train_, y_train_, batch_size=batch_size, epochs=epochs)

# model evaluation
print("\nTraining Set: ")
model2.evaluate(x_train_,  y_train_, verbose=2)
print("\nTest Set: ")
model2.evaluate(x_test_,  y_test_, verbose=2)


# plot 20 example images with predictions
num = 20
indices=random.sample(range(len(x_train)), num)
images = x_train[indices]
labels = y_train[indices]

probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

probabilities = probability_model(images)
#print(probabilities)
#print(tf.argmax(tf.nn.softmax(probabilities), 1))
predictions = tf.argmax(tf.nn.softmax(probabilities), 1).numpy()
#predictions



#num_row = 4
#num_col = 5
#fig, axes = plt.subplots(num_row, num_col, figsize=(1.5*num_col,2*num_row))
#for i in range(num):
#    ax = axes[i//num_col, i%num_col]
#    ax.imshow(images[i], cmap='gray_r')
#    ax.set_title('Label: {label},\nPrediction: {pred}'.format(label=labels[i], pred=predictions[i]))
#plt.tight_layout()
#plt.show()



# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.

test_labels=labels
test_images=images
predictions= probabilities

def plot_image(i, predictions_array, true_label, img):
  true_label, img = true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("Predicted: {}, p={:2.0f}% \n(Truth: {})".format(predicted_label,
                                100*np.max(predictions_array),
                                true_label),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

num_rows = 6
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()