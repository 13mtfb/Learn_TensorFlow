# tensorflow and tf.keras
import tensorflow as tf
from tensorflow import keras

# helper libraries
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# import the MNIST fashion dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# normalize the greyscale image
x_train, x_test = x_train / 255.0,  x_test / 255.0

# save class names for later plotting
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
                'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# explore the data
#print ("Test set:")
#print("x_train shape: "+ str(x_train.shape))
#print("y_train shape: " + str(y_train.shape))
#print("\nTrain set:")
#print("x_test shape: "+ str(x_test.shape))
#print("y_test shape: " + str(y_test.shape))

#plot the first train image
# plt.figure()
# plt.imshow(x_train[0])
# plt.colorbar()
# plt.grid(False)
# plt.show()

# plot the first 25 images from the training set
#plt.figure(figsize=(10,10))
#for i in range(25):
#    plt.subplot(5,5,i+1)
#    plt.xticks([])
#    plt.yticks([])
#    plt.grid(False)
#    plt.imshow(x_train[i], cmap=plt.cm.binary)
#    plt.xlabel(class_names[y_train[i]])
#plt.show()

# create the neural network model using the Sequential class
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

# create a custom optimizer
optimizer = tf.keras.optimizers.Adam()

# complie the model
model.compile(optimizer=optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

# train the model
model.fit(x=x_train, y=y_train, epochs=10)

# test the models
model.evaluate(x=x_test, y=y_test, verbose = 2)

# attach a softmax to the output to convert the logits to probabilities
probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
predictions = probability_model.predict(x_test)


#print("predictions for first image:" + str(class_names[np.argmax(predictions[0])]))
# plot the first test image
#plt.figure()
#plt.imshow(x_test[0])
#plt.colorbar()
#plt.grid(False)
#plt.show()

def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array, true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                        100*np.max(predictions_array),
                                        class_names[true_label]),
                                        color=color)

def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array, true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], y_test, x_test)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], y_test)
plt.tight_layout()
plt.show()
