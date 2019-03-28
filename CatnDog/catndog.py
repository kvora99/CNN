from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Dropout
from keras.layers.normalization import BatchNormalization
from keras import regularizers
import os
from matplotlib import pyplot as plt
import cv2
from keras.models import load_model
import numpy as np
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


data_gen = ImageDataGenerator(preprocessing_function=preprocess_input, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)

train_gen = data_gen.flow_from_directory('/home/bg22/PycharmProjects/KUNAL/dataset/training_set', target_size=(64,64), batch_size=16, class_mode='categorical')
test_gen = data_gen.flow_from_directory('/home/bg22/PycharmProjects/KUNAL/dataset/test_set', target_size=(64,64), batch_size=16, class_mode='categorical')

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(BatchNormalization())
#model.add(Dropout(0.5))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
#model.add(Dropout(0.5))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(16, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
#model.add(Dropout(0.5))
model.add(MaxPool2D(pool_size=(2, 2)))

model.add(Conv2D(16, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
#model.add(Dropout(0.5))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())

#model.add(Dense(16, activation='relu'))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit_generator(train_gen, epochs=20, validation_data=test_gen)


model.save('catndog.h5')


# # summarize history for accuracy
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.savefig('accuracy.png')
# plt.show()
#
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.savefig('loss.png')
# plt.show()
