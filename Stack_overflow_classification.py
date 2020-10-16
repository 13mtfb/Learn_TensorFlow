# code for tutorial from tensorflow website available here:
#   https://www.tensorflow.org/tutorials/keras/text_classification
#
# Exercise: Multiclass classification on Stack Overflow questions
#

# import libaries
import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

# ensure good version of tensorflow - coded at 2.3
print (tf.__version__)

#########################################################################
#################### CREATE THE DATASET
#########################################################################

# download and extract the stack overflow dataset
url = "https://storage.googleapis.com/download.tensorflow.org/data/stack_overflow_16k.tar.gz"

# create dataset
dataset = tf.keras.utils.get_file("stack_overflow_16k.tar.gz", url,
                                    untar=True, cache_dir='.',
                                    cache_subdir='')

#dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')

# print different parts of the directory
# print(os.listdir(dataset_dir))
#
# train_dir = os.path.join(dataset_dir, 'train')
# print(os.listdir(train_dir))

# prepare the dataset for binary classification using the text_dataset_from_directory utility
#   creates two folders on disk class_a (positive reviews), and class_b (negative reviews)
#   also split up the train set into 80:20 train/validation

# delete additional folders not needed
#remove_dir = os.path.join(train_dir, 'unsup')
#shutil.rmtree(remove_dir)

batch_size = 32
seed = 42

#train tf.Data.dataset object
raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed)

#validation tf.Data.dataset object
raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'train',
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=seed)

#test tf.Data.dataset object
raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
    'test',
    batch_size=batch_size)


# # print out a few examples of raw_train_ds which is a tf.Data object
# for text_batch, label_batch in raw_train_ds.take(1):
#     for i in range(3):
#         print("Review", text_batch.numpy()[i])
#         print("Label", label_batch.numpy()[i])
#
# print out class names from the tf.Data object
print("Label 0 corresponds to", raw_train_ds.class_names[0])
print("Label 1 corresponds to", raw_train_ds.class_names[1])
print("Label 2 corresponds to", raw_train_ds.class_names[2])
print("Label 3 corresponds to", raw_train_ds.class_names[3])

#########################################################################
#################### PRE-PROCESS THE DATASET
#########################################################################

# standardize, tokenize, and vectorize the data using the helpful preprocessing.TextVectorization layer.

# Standardization refers to preprocessing the text, typically to remove punctuation or HTML elements to
# simplify the dataset. Tokenization refers to splitting strings into tokens (for example, splitting a
# sentence into individual words, by splitting on whitespace). Vectorization refers to converting tokens
# into numbers so they can be fed into a neural network. All of these tasks can be accomplished with this layer.

# Note Standardization does not by default remove HTML from the reviews. Let's write a custom method to do that
def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation),
                                    '')

# Next create a TextVectorization layer to standardize, tokenize, and vectorize our data
max_features = 5000
sequence_length = 500

vectorize_layer = TextVectorization(
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

# make a text-only dataset (without labels), then call adapt
# adapt will fit the state of the preprocessing layer to the dataset
# this will cause the model to build an index of strings to integers
train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)


# create a function to see the result of using this layer to preprocess some data
def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label

# retrieve a batch (of 32 reviews annd labels) from the dataset
# text_batch, label_batch = next(iter(raw_train_ds))
# first_review, first_label = text_batch[0], label_batch[0]
# print("Review", first_review)
# print("Label", raw_train_ds.class_names[first_label])
# print("Vectorized review", vectorize_text(first_review, first_label))
# you can see that each token (word) has been replaced by an integer.
# you ccan lookup the token(string) that each integer corresponds to by calling
# .get_vocabulary() on the layer
# print("1287 ---> ",vectorize_layer.get_vocabulary()[1287])
# print(" 313 ---> ",vectorize_layer.get_vocabulary()[313])
# print('Vocabulary size: {}'.format(len(vectorize_layer.get_vocabulary())))


# final preprocessing step is to apply the TextVectorization layer to the train,
# test and validation dataset
train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)


#########################################################################
#################### CONFIGURE DATASET FOR PERFORMANCE
#########################################################################

# These are two important methods you should use when loading data to make sure that I/O does not become blocking.
#
#     .cache() keeps data in memory after it's loaded off disk. This will ensure the dataset does not become a
#     bottleneck while training your model. If your dataset is too large to fit into memory, you can also use
#     this method to create a performant on-disk cache, which is more efficient to read than many small files.
#
#     .prefetch() overlaps data preprocessing and model execution while training.

AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)


#########################################################################
#################### CREATE THE MODEL
#########################################################################

embedding_dim = 128

model = tf.keras.Sequential([
    layers.Embedding(max_features + 1, embedding_dim),
    layers.Dropout(0.2),
    layers.GlobalAveragePooling1D(),
    layers.Dropout(0.2),
    layers.Dense(4)])

# print a summary of the model
model.summary()


#The layers are stacked sequentially to build the classifier:
#   The first layer is an Embedding layer. This layer takes the integer-encoded reviews
#   and looks up an embedding vector for each word-index. These vectors are learned as
#   the model trains. The vectors add a dimension to the output array. The resulting
#   dimensions are: (batch, sequence, embedding).
#
#   Next, a GlobalAveragePooling1D layer returns a fixed-length output vector for each
#   example by averaging over the sequence dimension. This allows the model to handle
#   input of variable length, in the simplest way possible.
#
#   This fixed-length output vector is piped through a fully-connected (Dense) layer
#   with 16 hidden units.
#
#   The last layer is densely connected with a single output node.

# Loss function and optimizer
model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy']
)


#########################################################################
#################### TRAIN THE MODEL
#########################################################################

# Train the model
epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs)


#########################################################################
#################### EVALUATE THE MODEL
#########################################################################

loss, accuracy = model.evaluate(test_ds)

print("Loss: ", loss)
print("Accuray: ", accuracy)


#########################################################################
#################### PLOT ACCURACY AND LOSS
#########################################################################

# model.fit() returns a History object that contains a dictionary with everything
# that happened during training

history_dict = history.history
history_dict.keys()

acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)

# "bo" is for "blue dot"
plt.plot(epochs, loss, 'bo', label='Training loss')
# b is for "solid blue line"
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()


plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')

plt.show()

#########################################################################
#################### EXPORT THE MODEL
#########################################################################

export_model = tf.keras.Sequential([
    vectorize_layer,
    model,
    layers.Activation('sigmoid')
])

export_model.compile(
    loss=losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy']
)

# test with raw_test_ds which yields raw strings
loss, accuracy = export_model.evaluate(raw_test_ds)
print("export_model: ", accuracy)

#########################################################################
#################### PRINT SOME EXAMPLES
#########################################################################

for example, label in raw_test_ds.take(1):
    print("\n\nEXAMPLE:")
    print(example.numpy()[0])
    print("\n\nLABEL:")
    print(label.numpy()[1])


predictions = export_model.predict(example[0])
print(predictions)
