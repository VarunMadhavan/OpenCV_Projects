import os
import caer
import canaro
import numpy as np
import cv2 as cv
import gc
# import matplotlib.pyplot as plt


IMG_SIZE = (80,80)
channels = 1
char_path = 'simpsons_dataset'

char_dict = {}
for char in os.listdir(char_path):
    char_dict[char] = len(os.listdir(os.path.join(char_path,char)))

# Sort dict in descending order
char_dict = caer.sort_dict(char_dict, descending=True)
# print(char_dict)

characters=[]
count = 0
for i in char_dict:
    characters.append(i[0])
    count += 1
    if count >=10: break

# print(characters)

# Create training data
train = caer.preprocess_from_dir(char_path, characters, channels=channels, IMG_SIZE=IMG_SIZE, isShuffle=True)


# print(len(train))
featureSet, labels = caer.sep_train(train, IMG_SIZE=IMG_SIZE)

# Normalize featureSet between 0 and 1
featureSet = caer.normalize(featureSet)

# One hot encoding labels
from tensorflow.keras.utils import to_categorical
labels = to_categorical(labels, len(characters))

x_train, x_val, y_train, y_val = caer.train_val_split(featureSet, labels, val_ratio=0.2)
del train
del featureSet
del labels
gc.collect()


BATCH_SIZE = 32
Epochs = 10
# Image data generator
datagen = canaro.generators.imageDataGenerator()
train_gen = datagen.flow(x_train, y_train, batch_size=BATCH_SIZE)


# Creating the model
model = canaro.models.createSimpsonsModel(IMG_SIZE=IMG_SIZE, channels=channels, output_dim=len(characters),
                                          loss='binary_crossentropy', decay=1e-6, learning_rate=0.001,
                                          momentum=0.9, nesterov=True)

model.summary()

# Call back scheduler
from tensorflow.keras.callbacks import LearningRateScheduler
callbacks_list=[LearningRateScheduler(canaro.lr_schedule)]

training = model.fit(train_gen,steps_per_epoch=len(x_train)//BATCH_SIZE,
                     epochs=Epochs,validation_data=(x_val, y_val),
                     validation_steps=len(y_val)//BATCH_SIZE,
                     callbacks=callbacks_list)


test_paths = ["simpsons_dataset/bart_simpson/pic_0002.jpg","simpsons_dataset/bart_simpson/pic_0001.jpg","simpsons_dataset/bart_simpson/pic_0012.jpg","simpsons_dataset/bart_simpson/pic_0022.jpg","simpsons_dataset/lisa_simpson/pic_0002.jpg","simpsons_dataset/lisa_simpson/pic_0001.jpg","simpsons_dataset/lisa_simpson/pic_0012.jpg","simpsons_dataset/lisa_simpson/pic_0022.jpg"]


def prepare(img):
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, IMG_SIZE)
    img = caer.reshape(img, IMG_SIZE, 1)
    return img
def predicts(test_paths):
    for test_path in test_paths:
        img = cv.imread(test_path)
        prediction = model.predict(prepare(img))
        cv.putText(img, characters[np.argmax(prediction[0])], (25,25), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,255,0), thickness=1)

        cv.imshow(img)

predicts(test_paths)
cv.waitKey(0)
