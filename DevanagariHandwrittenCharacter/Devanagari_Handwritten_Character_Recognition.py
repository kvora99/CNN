from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.applications.inception_v3 import preprocess_input
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.python.keras import backend as k
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt

sz = 32
train_dir = 'DevanagariHandwrittenCharacterDataset/Train'
test_dir = 'DevanagariHandwrittenCharacterDataset/Test'

if k.image_data_format() == 'channels_first':
    input_shape = (3, sz, sz)
else:
    input_shape = (sz, sz, 3)

datagen = ImageDataGenerator(preprocessing_function=preprocess_input, shear_range=0.2, zoom_range=0.2)

train_gen = datagen.flow_from_directory(train_dir, target_size=(sz, sz), batch_size=16, class_mode='categorical')
test_gen = datagen.flow_from_directory(test_dir, target_size=(sz, sz), batch_size=16, class_mode='categorical')

label_map = train_gen.class_indices
print(label_map)

label_map1 = test_gen.class_indices
print(label_map1)

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
# model.add(Dense(64))
# model.add(Activation('softmax'))
# model.add(Dropout(0.5))
model.add(Dense(46))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit_generator(train_gen, epochs=10, validation_data=test_gen)

model.save('DevanagariHandwrittenCharacterRecognition.h5')


# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('accuracy.png')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('loss.png')
plt.show()
