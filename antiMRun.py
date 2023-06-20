import numpy as np 
import tensorflow as tf
import os # Importing default libs, np - arrays, tf - framework for AI itself, os - to work with folders, files and etc.

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model # importing methods and algos for AI

def numOfImgs(dir_path): #number of images in images folder
	count = 0
	for path in os.listdir(dir_path):
		if os.path.isfile(os.path.join(dir_path, path)): # check if current path is a file
			count += 1
	return count

def resizeImg(img, label): # resizing img for AI to work with it correctly
	img = tf.cast(img, tf.float32)
	img = tf.image.resize(img, (224, 224))
	img = img/225
	return img, label

s = os.getcwd().replace('/', '\\')

model = load_model(f"{s}\\antiM\\")


def run():
	num = numOfImgs(f"{s}\\images")
	try:
		os.mkdir(f"{s}\\infected")
		os.mkdir(f"{s}\\not_infected")
	except Exception as err:
		print(f"An error has occured [!] at mkdir: {err}")

	for i in range(num):
		try:
			img = load_img(f"{s}\\images\\{i+1}.jpg")
		except Exception as err:
			print(f"An error has occured [!] at imgload: {err}")



		imgArray = img_to_array(img)
		imgResized, img = resizeImg(imgArray, img)
		imgExpended = np.expand_dims(imgResized, axis=0)

		prediction = model.predict(imgExpended)[0][0]
		confidence = "NOT confident: " if prediction > -2 else "confident"
		infected_or_not = "infected" if prediction < 0.5 else "not_infected"
		plt.figure()
		plt.imshow(img)
		plt.title(f"{infected_or_not}, {confidence}: {prediction}")

		plt.savefig(f"{s}\\{infected_or_not}\\{i}.png")

run()
input()
"""
tensorflow - Google
keras - https://keras.io/
plt - https://matplotlib.org/
numpy - https://numpy.org/
"""