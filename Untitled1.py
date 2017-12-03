
# coding: utf-8

# In[4]:


import csv
import os
import sklearn
import cv2
import numpy as np
from random import shuffle
from IPython.core.debugger import Tracer

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.models import Model
import matplotlib.pyplot as plt

lines=[]
with open('./data/driving_log.csv') as csvfile:
    reader=csv.reader(csvfile)
    firstline=True
    for line in reader:
        if firstline:    #skip first line
            firstline = False
            #Tracer()()
            continue
        lines.append(line)
        
plt.hist(steering_ang)
plt.show()
plt.xlabel("Steering angle")
plt.ylabel("Number of data sets")
        
def gen_data(lines):
        
    images_path=[]
    measurements=[]
    correction=0.2
    for line in lines:
        center_images_path='./data/IMG/'+line[0].split('/')[-1]
        left_images_path='./data/IMG/'+line[1].split('/')[-1]
        right_images_path='./data/IMG/'+line[2].split('/')[-1]
        center_angle=float(line[3])
        left_angle=center_angle+correction
        right_angle=center_angle-correction
        images_path.extend((center_images_path,left_images_path,right_images_path))
        measurements.extend((center_angle,left_angle,right_angle))
    
    return images_path,measurements    
    

def generator(samples,batch_size=32):
    num_samples=len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0,num_samples,batch_size):
            batch_samples=samples[offset:offset+batch_size]
            
            images=[]
            angles=[]
            for images_path,measurements in batch_samples:
                image=cv2.cvtColor(cv2.imread(images_path), cv2.COLOR_BGR2RGB)
                images.append(image)
                angles.append(measurements)
                
            inputs=np.array(images)
            outputs=np.array(angles)
            yield sklearn.utils.shuffle(inputs, outputs)
            
images_path,measurements=gen_data(lines)
total_data=list(zip(images_path,measurements))

plt.hist(measurements)
plt.show()
plt.xlabel("Steering angle")
plt.ylabel("Number of data sets")
            
from sklearn.model_selection import train_test_split
train_samples, validation_samples=train_test_split(total_data,test_size=0.2)

print('Train samples: {}'.format(len(train_samples)))
print('Validation samples: {}'.format(len(validation_samples)))
            
train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32) 

ch, row, col = 3, 160, 320

model=Sequential()
model.add(Lambda(lambda x: (x / 255) - 0.5, input_shape=(row, col,ch),output_shape=( row, col,ch)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Convolution2D(64,3,3,activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(1))

model.compile(loss='mse',optimizer='adam')
history_object= model.fit_generator(train_generator, samples_per_epoch= len(train_samples), validation_data=validation_generator, 
            nb_val_samples=len(validation_samples), nb_epoch=3,verbose=1)

print(history_object.history.keys())

model.save('model.h5')
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
print("over")

