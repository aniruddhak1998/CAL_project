from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split


ydf = pd.read_csv('D:\Project\Code\sigData.csv')
is_Hazard = ydf['Affordance'] == 'Hazard'
y_Hazard = ydf[is_Hazard]
index = np.arange(0, y_Hazard.shape[0], 1)
y_Hazard = y_Hazard.set_index(index, 'Image No.')

def label_head(value):
    if value == 'FALSE':
        return 0
    else:
        return 1

label = y_Hazard['Ground_Truth_Affordance']
n = []
for i in range(len(label)):
    y_label = label_head(label[i])
    n.append(y_label)
label = n

model = VGG16(weights='imagenet', include_top=False)

images = []
for i in range(423):
    image = load_img('D:\Project\CameraRGB\image_00' + str(f'{i:03}') + '.jpg.png', target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)
    images.append(model.predict(image).reshape(7*7*512))

X_train, X_test, y_train, y_test = train_test_split(images, label, test_size=0.3)

learning_rate = 0.00005
n_neurons = 160
n_steps = 7
n_inputs = 7*512
n_outputs = 2
dropout = 0.68

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.int64, [None])

cell = tf.contrib.rnn.LSTMCell(num_units=n_neurons)
cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=1.0-dropout)
cell = tf.contrib.rnn.OutputProjectionWrapper(cell, output_size=2)
outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

logits = tf.layers.dense(inputs=states[1],units=n_outputs)
entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
loss = tf.reduce_mean(entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train = optimizer.minimize(loss)

correct = tf.nn.in_top_k(logits, y, 1)
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

n_epochs = 10000
batch_size = 8
init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()
    for n_epoch in range(n_epochs):
        i = 0
        print(n_epoch)
        while i < np.shape(images)[0] and i!=296:
            start = i
            end = i + batch_size
            i = i + batch_size
            X_batch = X_train[start:end]
            X_batch = np.reshape(X_batch, [batch_size, n_steps, n_inputs])
            y_batch = y_train[start:end]
            sess.run(train, feed_dict={X: X_batch, y: y_batch})
    print(sess.run(accuracy, feed_dict={X: np.reshape(X_test, [np.shape(X_test)[0], n_steps, n_inputs]), y: y_test}))