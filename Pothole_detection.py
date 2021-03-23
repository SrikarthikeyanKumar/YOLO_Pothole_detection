import cv2
import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import tensorflow as tf
import keras
from drawboundingbox import boundingbox
from Batchgenerator import batch_gen
from Yolo_models import YOLO_FULL_MODEL
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from yololoss import customloss

train_path='train\\'
train_datagen=batch_gen(train_path,8,416)
validation_path='validation\\'
val_datagen=batch_gen(validation_path,8,416)


'''
#Check y_true format
for i in range(0,4):l
    boundingbox(x[i,...], y[i,...])
'''

model=YOLO_FULL_MODEL()
#Trained weights for prediction
model.load_weights('best_weights_new1.h5',by_name=True)

'''
#For training
model.load_weights('yolov2-coco-original.h5',by_name=True)

optimizer=keras.optimizers.Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-8,decay=0.0005)

model.compile(optimizer=optimizer,loss=customloss)
early_stop=EarlyStopping(monitor='val_loss',min_delta=0.001,patience=6,mode='min',verbose=1)
checkpoint=ModelCheckpoint('best_weights_new1.h5',verbose=1,save_best_only=True,
                           mode='min',save_freq='epoch')
tensorboard=TensorBoard(log_dir='logs\\',histogram_freq=0)


model.fit(train_datagen, steps_per_epoch=6, epochs=80,
                    validation_data=val_datagen, validation_steps=4,
                    verbose=2,callbacks=[early_stop,checkpoint,tensorboard],
                   workers=1)
'''
count=0
for k in range(0,3):
    x1,y1=next(train_datagen)
    y_new=model.predict(x1)
    for i in range(0,8):
        boundingbox(x1[i,...], y_new[i,...],count,0.45,0.5)
        count+=1