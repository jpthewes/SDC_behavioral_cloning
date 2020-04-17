import cv2
import os
import csv
import cv2
import numpy as np
import sklearn
from math import ceil
from keras.models import Sequential, Model
from keras.layers import Cropping2D, Flatten, Dense, Lambda, Convolution2D, Dropout
from sklearn.model_selection import train_test_split
from scipy import ndimage


DATAPATH = '/home/workspace/full_round/'

# read in driving_log
samples = []
with open(DATAPATH + 'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
samples = samples[1:] # skip first line
print("read in SAMPLES")
     
# samle into training and validation set (20% is validation set)
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# use generator to get data on the fly while training the model
def generator(samples, batch_size=32):
    num_samples = len(samples)
    batch_size = int(batch_size/3) # for 3 pictures(left, right, center) each iteration, therefore roughly keeping batch size
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                center_name = os.path.join(DATAPATH , './IMG/' , batch_sample[0].split('/')[-1])
                center_image = ndimage.imread(center_name)
                center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)
                
                right_name = os.path.join(DATAPATH , './IMG/' , batch_sample[2].split('/')[-1])
                right_image = ndimage.imread(right_name)
                right_angle = float(batch_sample[3]) - 0.2 # 0.2 degrees correction
                images.append(right_image)
                angles.append(right_angle)
                
                left_name = os.path.join(DATAPATH , './IMG/' , batch_sample[1].split('/')[-1])
                left_image = ndimage.imread(left_name)
                left_angle = float(batch_sample[3]) + 0.2 # 0.2 degrees correction
                images.append(left_image)
                angles.append(left_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set batch size
batch_size=32

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

## model layout:
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320,3))) # normalize input
model.add(Cropping2D(cropping=((60,20),(0,0)))) # crop input images to relevant fiel of view
model.add(Convolution2D(24, kernel_size=(5,5), strides=(2,2), activation="relu"))
model.add(Convolution2D(36, kernel_size=(5,5), strides=(2,2), activation="relu"))
#model.add(Dropout(0.5)) # These have been shown to decrease driving performance
model.add(Convolution2D(48, kernel_size=(5,5), strides=(2,2), activation="relu"))
#model.add(Dropout(0.5)) # These have been shown to decrease driving performance
model.add(Convolution2D(64, kernel_size=(3,3), activation="relu"))
model.add(Convolution2D(64, kernel_size=(3,3), activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1)) # 1 output as 1 param is to be predicted (steering angle)
print(model.summary())

###Compile and train model
model.compile(loss='mse', optimizer='adam')
print("compiled")
model.fit_generator(train_generator, 
            steps_per_epoch=ceil(len(train_samples)/batch_size), 
            validation_data=validation_generator, 
            validation_steps=ceil(len(validation_samples)/batch_size), 
            epochs=4, verbose=1)
model.save('my_loop_model.h5')
print("saved the model")