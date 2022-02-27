import numpy as np
import tensorflow as tf
import cv2
import os
from glob2 import glob
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Softmax, Activation, Dense
import pickle
from tensorflow.keras.datasets import mnist
from sklearn.metrics import recall_score, f1_score, precision_score
from train_CNN import read_data
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# data = mnist.load_data()
# (x_train, y_train), (x_test, y_test) = data
# x_train = x_train.reshape(-1, 28, 28, 1)
# x_test = x_test.reshape(-1, 28, 28, 1)
# y_train = tf.compat.v1.keras.utils.to_categorical(y_train, 10)
# y_test = tf.compat.v1.keras.utils.to_categorical(y_test, 10)

# with open('./export_models/test.pkl', mode='wb') as f:
#     pickle.dump({'x_train': x_train, 'y_train': y_train,
#                 'x_test': x_test, 'y_test': y_test}, f)
# with open('./export_models/test.pkl', mode='rb') as f:
#     data = pickle.load(f)
# y_train = data['y_train']
# print(y_train[:3])
x_train, x_test, y_train, y_test = read_data()


def VGG(X, Y):
    model = Sequential()

    # layer_1
    model.add(Conv2D(64, (3, 3), strides=(
        1, 1), input_shape=X[1:], padding='same', data_format='channels_last', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(64, (3, 3), strides=(1, 1), padding='same',
              data_format='channels_last', kernel_initializer='uniform', activation='relu'))
    model.add(MaxPooling2D((2, 2), padding='SAME'))

    # layer_2
    model.add(Conv2D(128, (3, 3), strides=(1, 1), padding='same',
              data_format='channels_last', activation='relu', kernel_initializer='uniform'))
    model.add(Conv2D(128, (2, 2), strides=(1, 1), padding='same',
              data_format='channels_last', activation='relu', kernel_initializer='uniform'))
    model.add(MaxPooling2D((2, 2), padding='SAME'))

    # layer_3
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same',
              data_format='channels_last', activation='relu'))
    model.add(Conv2D(256, (3, 3), strides=(1, 1), padding='same',
              data_format='channels_last', activation='relu'))
    model.add(Conv2D(256, (1, 1), strides=(1, 1), padding='same',
              data_format='channels_last', activation='relu'))
    model.add(MaxPooling2D((2, 2), padding='SAME'))
    # layer_4
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same',
              data_format='channels_last', activation='relu'))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same',
              data_format='channels_last', activation='relu'))
    model.add(Conv2D(512, (1, 1), strides=(1, 1), padding='same',
              data_format='channels_last', activation='relu'))
    model.add(MaxPooling2D((2, 2), padding='SAME'))

    # layer_5
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same',
              data_format='channels_last', activation='relu'))
    model.add(Conv2D(512, (3, 3), strides=(1, 1), padding='same',
              data_format='channels_last', activation='relu'))
    model.add(Conv2D(512, (1, 1), strides=(1, 1), padding='same',
              data_format='channels_last', activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())  # 拉平
    model.add(Dense(4096, activation='relu'))
    # model.add(Dense(4096, activation='relu'))
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(4, activation='softmax'))

    model.summary()
    model.compile(optimizer='adam',
                  loss='mse', metrics=['accuracy'])

    return model


# x_train = data['x_train']
# y_train = data['y_train']
# x_test = data['x_test']
# y_test = data['y_test']
model = VGG((-1, 250, 250, 3), None)
x_train = np.array(x_train)
y_train = np.array(y_train)
x_test = np.array(x_test)
y_test = np.array(y_test)
model.fit(x=x_train, y=y_train, batch_size=28, epochs=20)
y_pred = model.predict(x_train)
model.save('vgg.h5')
# print(y_pred)
# tf = tf.compat.v1

# for cls_path in glob(os.path.join('./data', '*')):
#     for img_path in glob(os.path.join(cls_path, '*')):
#         # print(img_path)
#         img_path = img_path.replace('\\', '/')
#         print(img_path)
#         img = cv2.imread('./data/正视/0.jpg')
#         cv2.imshow('img', img)
#         cv2.waitKey(0)
# img = cv2.imread('./data/正视/1.jpg')

# cv2.imshow('test', img)
# cv2.waitKey(0)
