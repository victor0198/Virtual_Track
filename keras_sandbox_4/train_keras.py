from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.layers import Flatten, Dense, Activation, BatchNormalization, Conv2D, Dropout
from tensorflow.keras import backend as K
import numpy as np
from tensorflow.keras.preprocessing import image


img_width, img_height = 200, 66
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 4580
nb_validation_samples = 1320
epochs = 80
batch_size = 60

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)    

train_datagen = ImageDataGenerator(
    rescale = 1. / 255,
    zoom_range = 0.2)

test_datagen = ImageDataGenerator(
    rescale = 1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size = (img_width, img_height),
    batch_size = batch_size,
    class_mode = 'binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size = (img_width, img_height),
    batch_size = batch_size,
    class_mode = 'binary')

model = Sequential()
model.add(Conv2D(24, kernel_size=(5,5), strides=(2,2), input_shape = input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(36, kernel_size=(5,5), strides=(2,2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Conv2D(48, kernel_size=(3,3), strides=(2,2)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (1,1)))
model.add(Conv2D(64, kernel_size=(1,1), strides=(1,1)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (1,1)))

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(28))
model.add(Activation('softmax'))

model.summary()

model.compile(
    loss = 'sparse_categorical_crossentropy',
    optimizer = 'adam',
    metrics = ['accuracy'])

model.fit_generator(
    train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = epochs,
    validation_data = validation_generator,
    validation_steps = nb_validation_samples // batch_size)

model.save("saved_model_custom_12.h5")

model.save_weights('saved_model_weights_custom_12.h5')


