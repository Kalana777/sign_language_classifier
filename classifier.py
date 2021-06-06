import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from os import getcwd


def get_data(filename):
    with open(filename) as training_file:
        # Your code starts here
        csv_reader = csv.reader(training_file, delimiter=',')
        line_count = 0

        labels = []
        image_list = []
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
                pass
            else:
                labels.append(int(row[0]))
                k = np.array(row[1:785]).astype('int')
                j = np.array_split(k, 28)
                image_list.append(j)
                line_count += 1

        labels = np.array(labels)
        images = np.array(image_list).astype('int')
    # Your code ends here
    return images, labels


path_sign_mnist_train = f"{getcwd()}/data/sign_mnist_train.csv"
path_sign_mnist_test = f"{getcwd()}/data/sign_mnist_test.csv"
training_images, training_labels = get_data(path_sign_mnist_train)
testing_images, testing_labels = get_data(path_sign_mnist_test)

print(training_images.shape)
print(training_labels.shape)
print(testing_images.shape)
print(testing_labels.shape)

training_images = np.expand_dims(training_images, axis=3)
testing_images = np.expand_dims(testing_images, axis=3)


train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    #     horizontal_flip=True,
    fill_mode='nearest'
)
# from tensorflow.keras.utils import to_categorical

# training_labels = to_categorical(training_labels)
# testing_labels = to_categorical(testing_labels)
training_generator = train_datagen.flow(
    training_images,
    training_labels,
    batch_size=32
)

validation_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0
)
validation_generator = validation_datagen.flow(
    testing_images,
    testing_labels,
    batch_size=32
)

# Keep These
print(training_images.shape)
print(testing_images.shape)


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation = 'relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(25, activation='softmax')
])

# Compile Model.
model.compile(loss = 'sparse_categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])


# Train
history = model.fit_generator(training_generator, epochs=30, steps_per_epoch=len(training_images)/32, validation_data = validation_generator, verbose = 1, validation_steps=len(testing_images)/32)

model.evaluate(testing_images, testing_labels, verbose=0)

import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()