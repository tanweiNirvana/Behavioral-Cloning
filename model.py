import csv
import cv2
import numpy as np
import sklearn
from random import shuffle
#load data
lines = []
with open('data/driving_log.csv') as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)

#split train and validation
from sklearn.model_selection import train_test_split
train_lines, validation_lines = train_test_split(lines, test_size=0.2)

#generator:input: lines and batch_size
#output: image and angle
def generator(lines,batch_size = 16):
    num_lines = len(lines)
    while 1: # Loop forever so the generator never terminates
        #shuffle data
        shuffle(lines)
        for offset in range(0,num_lines,batch_size):#Divide the lines by batch_size
            batch_lines = lines[offset:offset+batch_size]
            images = []
            angles = []
            #load image and angle
            for batch_line in batch_lines:
                for j in range(2):
                    name = 'data/IMG/'+batch_line[0].split('\\')[-1]
                    originalImage = cv2.imread(name)
                    #Make colorspace the BGR into RGB
                    image = cv2.cvtColor(originalImage,cv2.COLOR_BGR2RGB)
                    angle = float(batch_line[3])
                    #According to j to load positive and negative picture
                    if j == 0:
                        images.append(image)
                        angles.append(angle)
                    else:
                        images.append(cv2.flip(image,1))
                        angles.append(angle*-1.0)

                    X_train = np.array(images)
                    y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_lines)
validation_generator = generator(validation_lines)

from keras.models import Sequential
from keras.layers import Flatten, Dense ,Lambda ,Cropping2D
from keras.layers.convolutional import Convolution2D

from keras.layers.pooling import MaxPooling2D
#Model Architecture
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5,input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5, subsample=(2,2),activation='relu'))
model.add(Convolution2D(36,5,5, subsample=(2,2),activation='relu'))
model.add(Convolution2D(48,5,5, subsample=(2,2),activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Convolution2D(64,3,3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
#Train
model.compile(loss='mse',optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(train_lines)*2, validation_data=validation_generator, nb_val_samples=len(validation_lines)*2,nb_epoch=5)

model.save('model.h5')
exit()
