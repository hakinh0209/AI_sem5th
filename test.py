import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
from os import listdir
import cv2
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import random
from keras.models import  load_model
import sys
from os import listdir
import cv2
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from keras.applications.vgg16 import VGG16
from keras.layers import Input, Flatten, Dense, Dropout
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
import random
from keras.preprocessing.image import ImageDataGenerator
import random
from keras.layers import Dense, Activation, Dropout, Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras import Sequential
class_name = [ 'A', 'B', 'C', 'D' , 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z' , 'del' , 'space']

def get_model():
    my_model = Sequential()

    my_model.add(Conv2D(64, (3,3), activation="relu", padding="same", kernel_initializer='he_uniform', name='block1_conv1', input_shape=(128, 128, 3)))
    my_model.add(Conv2D(64, (3,3), activation="relu", padding="same", kernel_initializer='he_uniform', name='block1_conv2'))
    my_model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), name='block1_maxpool'))

    my_model.add(Conv2D(128, (3,3), activation="relu", padding="same", kernel_initializer='he_uniform', name='block2_conv1'))
    my_model.add(Conv2D(128, (3,3), activation="relu", padding="same", kernel_initializer='he_uniform', name='block2_conv2'))
    my_model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), name='block2_maxpool'))

    my_model.add(Conv2D(256, (3,3), activation="relu", padding="same", kernel_initializer='he_uniform', name='block3_conv1'))
    my_model.add(Conv2D(256, (3,3), activation="relu", padding="same", kernel_initializer='he_uniform', name='block3_conv2'))
    my_model.add(Conv2D(256, (3,3), activation="relu", padding="same", kernel_initializer='he_uniform', name='block3_conv3'))
    my_model.add(Conv2D(256, (3,3), activation="relu", padding="same", kernel_initializer='he_uniform', name='block3_conv4'))
    my_model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), name='block3_maxpool'))

    my_model.add(Conv2D(512, (3,3), activation="relu", padding="same", kernel_initializer='he_uniform', name='block4_conv1'))
    my_model.add(Conv2D(512, (3,3), activation="relu", padding="same", kernel_initializer='he_uniform', name='block4_conv2'))
    my_model.add(Conv2D(512, (3,3), activation="relu", padding="same", kernel_initializer='he_uniform', name='block4_conv3'))
    my_model.add(Conv2D(512, (3,3), activation="relu", padding="same", kernel_initializer='he_uniform', name='block4_conv4'))
    my_model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), name='block4_maxpool'))

    my_model.add(Conv2D(512, (3,3), activation="relu", padding="same", kernel_initializer='he_uniform', name='block5_conv1'))
    my_model.add(Conv2D(512, (3,3), activation="relu", padding="same", kernel_initializer='he_uniform', name='block5_conv2'))
    my_model.add(Conv2D(512, (3,3), activation="relu", padding="same", kernel_initializer='he_uniform', name='block5_conv3'))
    my_model.add(Conv2D(512, (3,3), activation="relu", padding="same", kernel_initializer='he_uniform', name='block5_conv4'))
    my_model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2), name='block5_maxpool'))

    my_model.add(Flatten())
    my_model.add(Dense(1024, activation='relu', name='fc1'))
    my_model.add(Dense(1024, activation='relu', name='fc2'))
    my_model.add(Dropout(0.5))


    # Compile
    my_model.add(Dense(28, activation='softmax', name='predictions'))

    my_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return my_model
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Load weights model da train
my_model = get_model()
my_model.load_weights(r"C:\Users\kinh\Downloads\weights-12-1.00.hdf5")


offset = 20
imgSize = 300

folder = "test"
counter = 0
text =""
start = time.time()
letter_old = ""
letter = ""
check_even= 0
while True:

    success, img = cap.read()
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        # imgcrop_ = cap[y - offset:y + h + offset, x - offset:x + w + offset]
        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            # imgResize_ = cv2.resize(imgcrop_, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            # imgResize_ = cv2.resize(imgcrop_, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize

        # cv2.imshow("ImageCrop", imgCrop)
        # cv2.imshow("ImageWhite", imgWhite)
        # cv2.imshow("Image", img)
        image = cv2.resize(imgWhite, dsize=(128, 128))
        image = image.astype('float')*1./255
        # Convert to tensor
        image = np.expand_dims(image, axis=0)

        # Predict
        predict = my_model.predict(image)
        # print("x = , y = ",x,y)
        # print("This picture is: ", class_name[np.argmax(predict[0])], (predict[0]))
        # letter_current = class_name[np.argmax(predict[0])]
        
        end = time.time()
        if((end - start >= 1)  and check_even == 0 ):
            start = time.time()
            letter  = class_name[np.argmax(predict[0])]
            check_even = 1 
        if ((end - start >= 1)  and check_even == 1):
            check_even =0 
            start = time.time()
            letter_old = class_name[np.argmax(predict[0])]
            if(letter_old == letter):
                if letter != "del" and letter != "space":
                    text += letter
                elif letter == "del" :
                    text = text[0:-1]
                else:
                    text += "_"
        cv2.putText(img, class_name[np.argmax(predict[0])], (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        


    cv2.putText(img, text, (40, 40), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 3)

    cv2.imshow("Image", img)
    
    # cv2.putText(img, "hahaha", (0, 0,-100), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
    # key = cv2.waitKey(1)

    key = cv2.waitKey(1)
    if key == ord("r"):
        text = ""

