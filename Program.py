import numpy as np
import pandas as pd
import os
import cv2 #OpenCV
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import random

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from keras.layers import Dense, Activation, Flatten, Dropout

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

#mount google drive to import dataset stored in the drive
from google.colab import drive
drive.mount('/content/drive')

fire_images_path = r"/content/drive/MyDrive/cv_firedetection/fire_dataset/fire_images"
non_fire_images_path = r"/content/drive/MyDrive/cv_firedetection/fire_dataset/non_fire_images"

fire = len(os.listdir(fire_images_path))
non_fire = len(os.listdir(non_fire_images_path))

count = pd.DataFrame([[fire,non_fire]],columns=['fire','non_fire'])
count.plot(y=['fire','non_fire'], kind="bar", figsize=(5, 5))
plt.show()

print('Count of Fire Images = ', fire)
print('Count of Fire Images = ', non_fire)

#Fire Images
for img in os.listdir(fire_images_path)[:5]:
  img_array = cv2.imread(os.path.join(fire_images_path,img))
  plt.imshow(img_array)
  plt.show()

#Non-Fire Images
for img in os.listdir(non_fire_images_path)[:5]:
  img_array = cv2.imread(os.path.join(non_fire_images_path,img))
  plt.imshow(img_array)
  plt.show()

#Training and testing the CNN model
img_array.shape

data = []
IMG_SIZE = 300

def get_data(file_path,class_num, data):
    for img in os.listdir(file_path):
        try:
            img_array = cv2.imread(os.path.join(file_path,img))
            new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
            data.append([new_array, class_num])
        except:
            pass
get_data(fire_images_path,1,data)
get_data(non_fire_images_path,0,data)

random.shuffle(data)
X = []
y = []

for i in data:
    X.append(i[0])
    y.append(i[1])
X = np.array(X).reshape(-1,IMG_SIZE, IMG_SIZE,3)
y = np.array(y)

X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2)
X_test,X_val,y_test,y_val = train_test_split(X_test,y_test,test_size=0.5)

print(X_train.shape)
print(X_test.shape)
print(X_val.shape)

model = models.Sequential([
    
  layers.Conv2D(32, kernel_size = (3,3), padding='same', activation='relu', input_shape=(IMG_SIZE,IMG_SIZE,3)),
  layers.MaxPooling2D(),
    
  layers.Conv2D(64, kernel_size = (3,3), padding='same', activation='relu'),
  layers.MaxPooling2D(),
    
  layers.Conv2D(64, kernel_size = (3,3), padding='same', activation='relu'),
  layers.MaxPooling2D(),

  layers.Conv2D(64, kernel_size = (3,3), padding='same', activation='relu'),
  layers.MaxPooling2D(),
    
  layers.Conv2D(64, kernel_size = (3,3), padding='same', activation='relu'),
  layers.MaxPooling2D(),
    
  layers.Flatten(),
  layers.Dense(60, activation='relu'),
  layers.Dense(1, activation='sigmoid')
])
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=20)

#visualize the training vs the validation accuracy of the model
plt.plot(history.history['accuracy'], c="b")
plt.plot(history.history['val_accuracy'], c="g")
plt.title('Training History')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.show()

model.evaluate(X_test,y_test)
predictions = model.predict(X_test)
     
predicted= []
for i in predictions:
    if i >0.5:
        predicted.append(1)
    else:
        predicted.append(0)
predicted[:5]

cm = tf.math.confusion_matrix(labels=y_test,predictions=predicted)
sns.heatmap(cm,annot=True, fmt='d')
plt.figure(figsize=(20,40))

cat = ["no fire", "fire"]

for i in range(20):
    plt.subplot(8,6,i+1)
    plt.imshow((X_test[i] * 255).astype(np.uint8))
    plt.xlabel(cat[predicted[i]])