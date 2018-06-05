import matplotlib.pyplot as plt
import random
import os
import pyttsx3
import cv2
import csv
import skimage
from skimage import data
import numpy as np
import tensorflow as tf
from skimage import transform
from skimage.color import rgb2gray

def load_data(TrafficSigns):
    directories = [d for d in os.listdir(TrafficSigns) 
                   if os.path.isdir(os.path.join(TrafficSigns, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(TrafficSigns, d)
        file_names = [os.path.join(label_directory, f) 
                      for f in os.listdir(label_directory) 
                      if f.endswith(".ppm")]
        for f in file_names:
            images.append(skimage.data.imread(f))
            labels.append(int(d))
    return images, labels

ROOT_PATH ="C:/Users/Shobit-PC/Desktop/INDIAN DATASET/"
train_data_directory = os.path.join(ROOT_PATH, "Dataset_Training")
test_data_directory = os.path.join(ROOT_PATH, "Dataset_Testing")
images, labels = load_data(train_data_directory)
images28 = [transform.resize(image, (28, 28)) for image in images]
images28 = np.array(images28)
images28 = rgb2gray(images28)
x = tf.placeholder(dtype = tf.float32, shape = [None, 28, 28])
y = tf.placeholder(dtype = tf.int32, shape = [None])
images_flat = tf.contrib.layers.flatten(x)
logits = tf.contrib.layers.fully_connected(images_flat, 19, tf.nn.relu)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, 
                                                                    logits = logits))
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)																	
correct_pred = tf.argmax(logits, 1)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
tf.set_random_seed(1234)
sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(2000):
        print('EPOCH', i)
        _, accuracy_val = sess.run([train_op, accuracy], feed_dict={x: images28, y: labels})
        if i % 10 == 0:
            print("Loss: ", loss)
        print('DONE WITH EPOCH')
		
img = cv2.imread(os.path.join(ROOT_PATH, "test.ppm"))
img = transform.resize(img, (28, 28))
img= rgb2gray(img)
img=img.reshape(1,28,28)
predicted = sess.run([correct_pred], feed_dict={x: img})[0]

filename=(os.path.join(ROOT_PATH, "Dataset_Training/info.csv"))
with open(filename, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if int(row['sno'])==predicted[0]:
            a=row['lbl']
			
engine = pyttsx3.init()
engine.say(a)
engine.runAndWait()
print(a)
			
			
