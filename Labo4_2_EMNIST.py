import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import keras

# USE THIS SHIT https://www.tensorflow.org/datasets/keras_example

# Load EMNIST !!! split into train and test here (different in keras)
(ds_train, ds_test), ds_info = tfds.load(
    'emnist/letters',
    split=['train', 'test'],
    shuffle_files=False,
    as_supervised=True,
    with_info=True
)

# Normalize each sample (why not just keras.normalize idk but hey ig tf-docs isn't wrong)
# ---------------------------------------------------------------------------------------
def normalize_img(image, label):
    return tf.cast(image, tf.float32) / 255.0, label

#apply normalize func on data batches
ds_train = ds_train.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)


#tf.data.Dataset.cache As you fit the dataset in memory, cache it before shuffling for a better *performance*.
        #Note: Random transformations should be applied after caching.

#tf.data.Dataset.shuffle: For true randomness, set the shuffle buffer to the full dataset size.
        #Note: For large datasets that can't fit in memory, use buffer_size=1000 if your system allows it.

#tf.data.Dataset.batch: Batch elements of the dataset after shuffling to get unique batches at each epoch.

#tf.data.Dataset.prefetch: It is good practice to end the pipeline by prefetching for performance.

#all by all to simplify: .cache().batch ARE REQUIRED!! Rest is performance
ds_train = ds_train.cache().shuffle(1000).batch(128).prefetch(tf.data.AUTOTUNE)
ds_test = ds_test.batch(128).cache().prefetch(tf.data.AUTOTUNE)
# ---------------------------------------------------------------------------------------


# View an example
for image, label in ds_train.take(1):
    print("LABELS")
    print(label)
    plt.imshow(image[0], cmap = plt.cm.binary) #Image is a BATCH of stuff (record of images), same goes for label (label of images) so image[0] combines with label[0]
    plt.show()


# model creations (look at Labo4_1 for explanations)
# ----------------------------------------------------------------------------------------
model = keras.Sequential()

model.add(tf.keras.layers.Flatten())
#model.add(keras.layers.Flatten() )

model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))

# LAST LAYER SIZE === MODEL OUTPUT AMOUNT +1 [1 - 26[
model.add(tf.keras.layers.Dense(27, activation=tf.nn.softmax))

model.compile (optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"] )

model.fit(ds_train, epochs = 3 )

val_loss, val_acc = model.evaluate(ds_test)

#start predict
prediction = model.predict( ds_train )
print(prediction)

print("argmax number")
print (np.argmax(prediction[0]))


def number_to_letter(data):
    res = chr(data+64) 
    return res

print("predicter letter")
print(number_to_letter(np.argmax(prediction[0])))



# -----------------------------------------------------------------------------------------