from unicodedata import name
import cv2
from sklearn import model_selection
import tensorflow as tf
import os
from glob2 import glob
from sklearn.model_selection import train_test_split
import numpy as np

# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
tf = tf.compat.v1
tf.disable_eager_execution()


def read_data():
    x_data = []
    y_data = []
    i = 0
    for cls_path in glob(os.path.join('./tmp_dataset', '*')):
        print(cls_path, i)
        # print(i)
        for img_path in glob(os.path.join(cls_path, '*')):
            # print(img_path)
            img = cv2.imread(img_path)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # print(img.shape)

            x_data.append(img)
            y_data.append(i)
        i += 1
    i = 0
    print('*'*100)
    for cls_path in glob(os.path.join('../../data/tmp_dataset2/tmp_dataset', '*')):
        print(cls_path, i)
        for img_path in glob(os.path.join(cls_path, '*')):
            # print(img_path)
            img = cv2.imread(img_path)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            img_path = img_path.replace('\\', '/')
            # print(img_path)
            if img is None:
                print('None')
                continue
            x_data.append(img)
            y_data.append(i)
        i += 1
    i = 0
    print('*'*100)

    for cls_path in glob(os.path.join('../../data/tmp_dataset3', '*')):
        print(cls_path, i)
        for img_path in glob(os.path.join(cls_path, '*')):
            img = cv2.imread(img_path)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            x_data.append(img)
            y_data.append(i)
        i += 1
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.2, random_state=10)

    # print(x_train[:10])
    # print(y_train[:10])
    y_train = tf.keras.utils.to_categorical(y_train, 4)
    y_test = tf.keras.utils.to_categorical(y_test, 4)

    return x_train, x_test, y_train, y_test
# print(y_train[:10])


def shuffle_data(data, label):
    idx = np.arange(len(data))
    np.random.shuffle(idx)
    return data[idx, ...], label[idx, ...]


def CNN2(data, label):

    #
    data_in = tf.placeholder(
        tf.float32, shape=[None, 250, 250, 3], name='data_in')
    label_in = tf.placeholder(tf.float32, shape=[None, 4])
    # layer1
    conv1 = tf.layers.conv2d(data_in, 64, (3, 3), strides=(
        1, 1), padding='SAME', activation=tf.nn.relu, kernel_initializer='uniform')
    conv1 = tf.layers.conv2d(conv1, 64, (3, 3), padding='SAME',
                             kernel_initializer='uniform', activation=tf.nn.relu)
    conv1 = tf.layers.max_pooling2d(
        conv1, (2, 2), strides=(1, 1), padding='SAME')

    # layer2
    conv2 = tf.layers.conv2d(conv1, 128, (3, 3), strides=(
        1, 1), padding='SAME', activation=tf.nn.relu)
    conv2 = tf.layers.conv2d(conv2, 128, (2, 2), padding='SAME',
                             kernel_initializer='uniform', activation=tf.nn.relu)
    conv2 = tf.layers.max_pooling2d(
        conv2, (2, 2), padding='SAME', strides=(1, 1))

    # layer3
    conv3 = tf.layers.conv2d(conv2, 256, (3, 3), strides=(
        1, 1), padding='SAME', activation=tf.nn.relu)
    conv3 = tf.layers.conv2d(conv3, 256, (3, 3), padding='SAME',
                             kernel_initializer='uniform', activation=tf.nn.relu)
    conv3 = tf.layers.conv2d(conv3, 256, (1, 1), padding='SAME',
                             kernel_initializer='uniform', activation=tf.nn.relu)
    conv3 = tf.layers.max_pooling2d(
        conv3, (2, 2), padding='SAME', strides=(1, 1))
    # layer4
    conv4 = tf.layers.conv2d(conv3, 512, (3, 3), padding='SAME',
                             kernel_initializer='uniform', activation=tf.nn.relu)
    conv4 = tf.layers.conv2d(conv4, 512, (3, 3), padding='SAME',
                             kernel_initializer='uniform', activation=tf.nn.relu)
    conv4 = tf.layers.conv2d(conv4, 512, (1, 1), padding='SAME',
                             kernel_initializer='uniform', activation=tf.nn.relu)
    conv4 = tf.layers.max_pooling2d(
        conv4, (2, 2), padding='SAME', strides=(1, 1))

    # layer5
    conv5 = tf.layers.conv2d(conv4, 512, (3, 3), padding='SAME',
                             kernel_initializer='uniform', activation=tf.nn.relu)
    conv5 = tf.layers.conv2d(conv5, 512, (3, 3), padding='SAME',
                             kernel_initializer='uniform', activation=tf.nn.relu)
    conv5 = tf.layers.conv2d(conv5, 512, (1, 1), padding='SAME',
                             kernel_initializer='uniform', activation=tf.nn.relu)
    conv5 = tf.layers.max_pooling2d(
        conv5, (2, 2), padding='SAME', strides=(1, 1))

    flatten = tf.layers.flatten(conv5)
    output = tf.layers.dense(flatten, 4096, activation=tf.nn.relu)
    output = tf.layers.dense(output, 1000, activation=tf.nn.relu)
    y_pred = tf.layers.dense(output, 4)
    outcome = tf.argmax(tf.nn.softmax(y_pred), 1, name='outcome')
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=y_pred, labels=label_in))
    opt = tf.train.AdamOptimizer(0.001).minimize(loss)
    # global_step = tf.Variable(0, trainable=False)
    # decaylearning_rate = tf.train.exponential_decay(
    #     0.0011, global_step, 1000, 0.9)
    # opt = tf.train.AdamOptimizer(decaylearning_rate).minimize(
    #     loss, global_step=global_step)
    content_acy = tf.equal(outcome, tf.argmax(label_in, 1))
    accuracy = tf.reduce_mean(tf.cast(content_acy, 'float'))
    batch_size = 16
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(100):
            total_loss = 0
            avg_accuracy = 0
            datas, labels = shuffle_data(data, label)
            num_batch = len(data)//batch_size
            for batch_idx in range(num_batch):
                start_idx = batch_idx*batch_size
                end_idx = (batch_idx+1)*batch_size
                feed_dict = {
                    data_in: datas[start_idx:end_idx, ...],
                    label_in: labels[start_idx:end_idx, ...]
                }
                _, cost, acy = sess.run(
                    [opt, loss, accuracy], feed_dict=feed_dict)
                total_loss += cost
                avg_accuracy += acy
            print('loss=', total_loss,
                  'avg_accuracy=', avg_accuracy/num_batch)
        saver.save(sess, './model')
        test_data = data[5*batch_size:6*batch_size, ...]

        test_label = label[5*batch_size:6*batch_size, ...]
        test_acy = sess.run(feed_dict={
            data_in: test_data,
            label_in: test_label
        })
        print('test_acy=', test_acy)


def CNN(x):
    model = tf.keras.Sequential()
    # model.build(input_shape=x.shape[1:])
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(
        3, 3), input_shape=x.shape[1:], padding='SAME', activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(
        3, 3), padding='SAME', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D((2, 2), padding='SAME', strides=2))

    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(
        3, 3), padding='SAME', activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(
        3, 3), padding='SAME', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D((2, 2), padding='SAME', strides=2))

    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(
        3, 3), padding='SAME', activation='relu'))
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(
        3, 3), padding='SAME', activation='relu'))
    model.add(tf.keras.layers.MaxPool2D((2, 2), padding='SAME', strides=2))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(1000, activation='relu'))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(4, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()
    return model
    pass


if __name__ == '__main__':
    x_train, x_test, y_train, y_test = read_data()
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    print(x_train.shape)
    print(y_train.shape)
    CNN2(x_train, y_train)
    # model.fit(x=x_train, y=y_train, batch_size=28, epochs=128)
    # model.save('./export_models/CNN.h5')

    # y_test = np.array(y_test)
    # x_test = np.array(x_test)
    # detect = tf.keras.models.load_model('./export_models/CNN.h5')
    # y_pred = detect.predict(x_test)
    # cnt = 0
    # outcome = (y_pred == y_test)
    # print(outcome)
    # for item in outcome:
    #     if item is True:
    #         cnt += 1
    # print(round(cnt/len(y_pred), 5))asd
