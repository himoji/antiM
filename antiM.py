import numpy as np 
import tensorflow as tf

import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model

imgSize = 224
def resizeImg(img, label):
	img = tf.cast(img, tf.float32)
	img = tf.image.resize(img, (imgSize, imgSize))
	img = img/225
	return img, label

train,  _ = tfds.load("malaria", split=['train[:100%]'], with_info=True, as_supervised=True) 
"""Training the AI with "malaria" dataset , author={Rajaraman, Sivaramakrishnan and Antani, Sameer K and Poostchi, Mahdieh
  and Silamut, Kamolrat and Hossain, Md A and Maude, Richard J and Jaeger,
  Stefan and Thoma, George R}, year={2018},
  publisher={PeerJ Inc.}

  """

trainResized = train[0].map(resizeImg)
trainBatches = trainResized.shuffle(1000).batch(16)

baseLayers = tf.keras.applications.MobileNetV2(input_shape=(imgSize, imgSize, 3), include_top=False) #The pre-gen AI,used to save time and resources
baseLayers.trainable = False

def create_model():
	model = tf.keras.Sequential([
								baseLayers,
								GlobalAveragePooling2D(),
								Dropout(0.2),
								Dense(1)
	])
	
	cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath="training_1/cp.ckpt",
	                                                 save_weights_only=True,
	                                                 verbose=1)
	
	model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=["accuracy"])
	return model

model = create_model()

model.fit(trainBatches, epochs=1)# [!] change num of epochs, to increase accuracy [!]

model.save('antiM')
"""
tensorflow - Google
keras - https://keras.io/
plt - https://matplotlib.org/
numpy - https://numpy.org/
"""