import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0


plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel(class_names[train_labels[i]])
plt.show()

model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(10, activation='softmax'))

model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=tf.keras.metrics.SparseCategoricalAccuracy())

model.fit(train_images, train_labels, batch_size=256, epochs=10)

test_loss, test_accu = model.evaluate(test_images, test_labels, verbose=2)
print('Test Accuracy:', test_accu)

predictions = model.predict(test_images)


def plot_image(i):
    img = test_images[i]
    predictions_array = predictions[i]
    predicted_label = np.argmax(predictions_array)
    true_label = test_labels[i]
    if predicted_label == true_label:
        color = 'b'
    else:
        color = 'r'
    plt.imshow(img, cmap=plt.cm.binary)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('{} {:2.0f}% ({})'.format(class_names[predicted_label], 100 * np.max(predictions_array),
                                         class_names[true_label]), color=color)


def plot_value_array(i):
    predictions_array = predictions[i]
    predicted_label = np.argmax(predictions_array)
    true_label = test_labels[i]
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color='#777777')
    plt.ylim([0, 1])
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


# plot_image(0)
# plot_value_array(0)
# plt.show()

model.save('fashion_model.h5')
