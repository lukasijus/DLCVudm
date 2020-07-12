import tensorflow as tf
import os
import math
import matplotlib.pyplot as plt
# load data
dir = os.path.dirname(os.path.abspath(__file__))
dir_train = os.path.join(dir, 'train')
dir_test = os.path.join(dir, 'test')
print('dir_train: ', dir_train)
print('dir_test', dir_test)
# prepare data
BATCH_SIZE = 100
IMAGE_HEIGHT = 45
IMAGE_WIDHT = 35
data = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_data = data.flow_from_directory(dir_train, target_size=(IMAGE_WIDHT, IMAGE_HEIGHT), batch_size=BATCH_SIZE, shuffle=True, class_mode='sparse')
test_data = data.flow_from_directory(dir_test, target_size=(IMAGE_WIDHT, IMAGE_HEIGHT), batch_size=BATCH_SIZE, shuffle=False, class_mode='sparse')
# compile model
INPUT_NEURONS = 512
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation=tf.nn.relu, input_shape=(IMAGE_WIDHT, IMAGE_HEIGHT, 3)),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D((2, 2), strides=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)

])
model.compile(optimizer='adam',loss = 'binary_crossentropy', metrics=['accuracy'])

EPOCH = 3
print(math.ceil(29980/BATCH_SIZE))
model.fit(train_data, epochs=EPOCH, steps_per_epoch=math.ceil(29980/BATCH_SIZE))












# save model

# Do predict