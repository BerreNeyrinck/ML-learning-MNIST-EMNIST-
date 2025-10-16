import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.datasets import mnist
import keras
# ./(venvname)/Scripts/activate


# cast data into training + testing containers
(x_train, y_train), (x_test , y_test) = mnist.load_data()

# # print visual represent of first X_train data cel
# plt.imshow(x_train[0], cmap = plt.cm.binary)
# plt.show()

# # normalize values 0-255 ~ 0-1
x_train = tf.keras.utils.normalize(x_train, axis=1)
# plt.imshow(x_train[0], cmap = plt.cm.binary)
# plt.show()

# print matrix with image pixel values
print(x_train[0])

# AI MODEL DECLARATION
# model = tf.keras.models.Sequential()
model = keras.Sequential()

#model compute variable "flatten" "converting the 4D array (m, height, width, channels) into a 2D array of shape (m, height × width × channels)" (some guy online)
# shortening this for ease of understanding, I suggest it :)
# model.add(tf.keras.layers.Flatten())
model.add(keras.layers.Flatten() )

#geeft ons model 128 parameters op LAYER: 1! 
#RELU (rectified Lineair Unit) in short: [if value > 0 = value, if value < 0 = 0]
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

#geeft ons model 128 parameters op LAYER: 2!
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

#geeft ons model 10 parameters op Layer: 3 LAATSTE LAAG!
#softmax makes a prediction with probabilities to create an output value!
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

#we kiezen hier de optimizer "adam", die gaat de keuze maken welke "weights" het beste zijn voor elke pathway in ons model.
#SPC (not writing that) is de loss-functie die we gebruiken om door onze paths tussen neurons te wandelen
model.compile (optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"] )

#model is nu bruikbaar, we fitten het onze training data en doen 3 loops
model.fit(x_train, y_train, epochs = 3 )

# ons model evalueren met testdata
val_loss, val_acc = model.evaluate(x_test, y_test)
# print(val_loss, val_acc)


#stel we willen onze AI nu gebruiken op degelijke data
# prediction = model.predict( <teBepalenElement> )
# !!! dit geeft niet super veel interresante info (% accuracy per nummer per foto)
# dus we gebruiken numpy om dit naar een definitieve waarde te zetten
# print (numpy.argmax(prediction))  

prediction = model.predict( x_train )
print (np.argmax(prediction[0]))