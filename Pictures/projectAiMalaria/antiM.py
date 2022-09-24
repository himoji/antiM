import numpy as np 
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model

checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

imgSize = 224 #Set img size
def resizeImg(img, label):
	img = tf.cast(img, tf.float32)
	img = tf.image.resize(img, (imgSize, imgSize))
	img = img/225
	return img, label

train,  _ = tfds.load("malaria", split=['train[:100%]'], with_info=True, as_supervised=True) # Loading dataset in ai
"""Training the AI with "malaria" dataset , author={Rajaraman, Sivaramakrishnan and Antani, Sameer K and Poostchi, Mahdieh
  and Silamut, Kamolrat and Hossain, Md A and Maude, Richard J and Jaeger,
  Stefan and Thoma, George R}, year={2018},
  publisher={PeerJ Inc.}

  """

trainResized = train[0].map(resizeImg)
trainBatches = trainResized.shuffle(1000).batch(16)

baseLayers = tf.keras.applications.MobileNetV2(input_shape=(imgSize, imgSize, 3), include_top=False) #The pre-gen AI,used to save time and resources
baseLayers.trainable = False

model = tf.keras.Sequential([
							baseLayers,
							GlobalAveragePooling2D(),
							Dropout(0.4),
							Dense(1)
])

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True), metrics=["accuracy"])
model.fit(trainBatches, epochs=1)# [!] change num of epochs, to increase accuracy [!]

model.save('my_model.h5')
dirName = os.path.dirname(os.path.abspath(__file__))
try:
	os.mkdir(f"{dirName}/infected")

except Exception as err:
	print("Failed make the folder")

for i in range(11):
	img = load_img(f"{dirName}/imgs/{i+1}.jpg")

	imgArray = img_to_array(img)
	imgResized, _ = resizeImg(imgArray, _)
	imgExpended = np.expand_dims(imgResized, axis=0)
	prediction = model.predict(imgExpended)[0][0]
	if prediction < 0.5:
		WhatIsLabel = "infected"
		plt.figure()
		plt.imshow(img)
		plt.title(f"{WhatIsLabel} {prediction}")
		plt.savefig(f"{dirName}/infected/{i+1}.jpg") #saves infected
		
#plt.show() #shows photos

"""
tensorflow - Google
keras - https://keras.io/
plt - https://matplotlib.org/
numpy - https://numpy.org/
"""