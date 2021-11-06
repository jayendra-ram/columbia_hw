#Generate dataset
import os
import pandas as pd
import numpy as np
from PIL import Image


#Load Training images and labels
train_directory = "./data/ecbm4040-assignment-2-task-5/kaggle_train_128/train_128" #TODO: Enter path for train128 folder (hint: use os.getcwd())
image_list=[]
label_list=[]
for sub_dir in os.listdir(train_directory):
    print("Reading folder {}".format(sub_dir))
    sub_dir_name=os.path.join(train_directory,sub_dir)
    for file in os.listdir(sub_dir_name):
        filename = os.fsdecode(file)
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_list.append(np.array(Image.open(os.path.join(sub_dir_name,file))))
            label_list.append(int(sub_dir))
X_train=np.array(image_list)
y_train=np.array(label_list)

#Load Test images
test_directory = "./data/ecbm4040-assignment-2-task-5/kaggle_test_128/test_128"#TODO: Enter path for test128 folder (hint: use os.getcwd())
test_image_list=[]
test_df = pd.DataFrame([], columns=['Id', 'X'])
print("Reading Test Images")
for file in os.listdir(test_directory):
    filename = os.fsdecode(file)
    if filename.endswith(".jpg") or filename.endswith(".png"):
        test_df = test_df.append({
            'Id': filename,
            'X': np.array(Image.open(os.path.join(test_directory,file)))
        }, ignore_index=True)
        
test_df['s'] = [int(x.split('.')[0]) for x in test_df['Id']]
test_df = test_df.sort_values(by=['s'])
test_df = test_df.drop(columns=['s'])
X_test = np.stack(test_df['X'])


print('Training data shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test data shape: ', X_test.shape)

import datetime
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from tensorflow.keras import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.losses import categorical_crossentropy
import datetime
from sklearn.model_selection import train_test_split

#X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

def create_model():
    model = tf.keras.models.Sequential()
    model.add(Conv2D(filters=32, kernel_size=(32,32), padding='same', activation='relu', input_shape=(128, 128, 3)))
    model.add(AveragePooling2D(strides=2))
    model.add(Conv2D(filters=64, kernel_size=(5,5), padding='valid', activation='relu'))
    model.add(MaxPooling2D(strides=2))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    #TODO: Add layers below
    
    
    return model

#Create the model, compile the model, and fit it
model_test = create_model()
model_test.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
model_test.fit(x=X_train, 
          y=y_train,
          batch_size=256,
          epochs=10, 
          callbacks=[tensorboard_callback])

model_test.save(filepath = "./model/task5_model" + datetime.datetime.now().strftime("%Y%m%d-%H"))
loaded_model = tf.keras.models.load_model("./model/task5_model" + datetime.datetime.now().strftime("%Y%m%d-%H"))
print(loaded_model.summary())
y_test = loaded_model.predict(X_test)

import csv
with open('predicted.csv','w') as csvfile:
    fieldnames = ['Id','label']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for index,l in enumerate(np.argmax(y_test, axis=1)):
        filename = str(index) + '.png'
        label = str(l)
        writer.writerow({'Id': filename, 'label': label})