import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras.preprocessing import image
from keras.optimizers import Adam
import numpy as np
from os import listdir
from os.path import isfile, join
from keras import backend as K
import cv2

def swish_activation(x):
    return (K.sigmoid(x) * x)

# dimensions of our images
img_width, img_height = 48, 48

# load the model we saved
# model = load_model('model-dllab.h5')
# model = tf.keras.models.load_model('fer2013_cnn.h5')
model = tf.keras.models.load_model('model-dllab.h5')
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
# model.compile(loss='categorical_crossentropy',
#               optimizer='rmsprop',
#               metrics=['accuracy'])


# my_path = "predict/"
my_path = "data/validation/Sad/"
only_files = [f for f in listdir(my_path) if isfile(join(my_path, f))]
print(only_files)

# predicting images
v1_counter = 0 
v2_counter = 0
v3_counter = 0
v4_counter = 0
v5_counter = 0
v6_counter = 0
v7_counter = 0

for file in only_files:
    img = image.load_img(my_path + file, target_size=(img_width, img_height))
    img = cv2.cvtColor(np.float32(img), cv2.COLOR_RGB2GRAY)
    x = image.img_to_array(img)
    x = x.reshape(48, 48, 1)
    x = np.expand_dims(x, axis=0)


    images = np.vstack([x])

    classes = model.predict_classes(images, batch_size=10)
    print('class: ', classes)
    # classes = classes[0][0]
    
    if classes == 0:
        print(file + ": " + 'Angry')
        v1_counter += 1
    elif classes == 1:
        print(file + ": " + 'Disgust')
        v2_counter += 1
    elif classes == 2:
        print(file + ": " + 'Fear')
        v3_counter += 1
    elif classes == 3:
        print(file + ": " + 'Happy')
        v4_counter += 1
    elif classes == 4:
        print(file + ": " + 'Sad')
        v5_counter += 1
    elif classes == 5:
        print(file + ": " + 'Surprise')
        v6_counter += 1
    elif classes == 6:
        print(file + ": " + 'Neutral')
        v7_counter += 1

print("Total Class 1 :", v1_counter)
print("Total Class 2 :", v2_counter)
print("Total Class 3 :", v3_counter)
print("Total Class 4 :", v4_counter)
print("Total Class 5 :", v5_counter)
print("Total Class 6 :", v6_counter)
print("Total Class 7 :", v7_counter)

