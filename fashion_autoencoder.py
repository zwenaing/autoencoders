import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from scipy.misc import imsave
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import homogeneity_score
from sklearn.cluster import KMeans

fashion = input_data.read_data_sets("data/fashion")
log_dir = "tmp/fashion_autoencoder"

# parameters
learning_rate = 0.01
num_steps = 10000
batch_size = 256
display_step = 100
examples_to_show = 10

# Network parameters
num_hidden_1 = 100
#num_hidden_2 = 128
num_input = 784

X = tf.placeholder(tf.float32, [None, num_input])

weights = {
    "encoder_w1": tf.Variable(tf.random_normal([num_input, num_hidden_1])),
    #"encoder_w2": tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
    "decoder_w1": tf.Variable(tf.random_normal([num_hidden_1, num_input])),
    #"decoder_w2": tf.Variable(tf.random_normal([num_hidden_1, num_input])),
    "encoder_b1": tf.Variable(tf.random_normal([num_hidden_1])),
    #"encoder_b2": tf.Variable(tf.random_normal([num_hidden_2])),
    "decoder_b1": tf.Variable(tf.random_normal([num_input])),
    #"decoder_b2": tf.Variable(tf.random_normal([num_input]))
}

def encoder(X):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(X, weights["encoder_w1"]), weights["encoder_b1"]))
    #layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights["encoder_w2"]), weights["encoder_b2"]))
    #return layer_2
    return layer_1

def decoder(encoded_X):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(encoded_X, weights["decoder_w1"]), weights["decoder_b1"]))
    #layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights["decoder_w2"]), weights["decoder_b2"]))
    #return layer_2
    return layer_1

with tf.name_scope("Encoding"):
    encode = encoder(X)

with tf.name_scope("Decoding"):
    decode = decoder(encode)

with tf.name_scope("Loss"):
    y_pred = decode
    y_true = X
    loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))

with tf.name_scope("Optimizing"):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    #saver.restore(sess, log_dir)
    for i in range(num_steps + 1):
       batch_x, _ = fashion.train.next_batch(batch_size)
       _, loss_ = sess.run([train_op, loss], feed_dict={X: batch_x})
       if i % display_step == 0:
           print("Step: ", i, " Loss: " + "{:.3f}".format(loss_))
    saver.save(sess, log_dir)

    train_encoded = sess.run(encode, feed_dict={X: fashion.train.images})
    #np.save("encoded_fashion_train.npy", train_encoded)
    test_encoded = sess.run(encode, feed_dict={X: fashion.test.images})
    #np.save("encoded_fashion_test.npy", test_encoded)

    clf = LogisticRegression()
    clf.fit(train_encoded, fashion.train.labels)
    print("Train Accuracy ", clf.score(train_encoded, fashion.train.labels))
    print("Train Accuracy ", clf.score(test_encoded, fashion.test.labels))

    clu = KMeans(n_clusters=10)
    clu.fit(train_encoded)
    print("Train purity score", homogeneity_score(fashion.train.labels, clu.predict(train_encoded)))
    print("Test purity score", homogeneity_score(fashion.test.labels, clu.predict(test_encoded)))

    n = 10
    canvas_original = np.empty((28 * n, 28 * 10))
    canvas_reconstruct = np.empty((28 * n, 28 * n))

    for i in range(n):
        batch_x, _ = fashion.test.next_batch(n)
        reconstruct_x = sess.run(decode, feed_dict={X: batch_x})

        for j in range(n):
            canvas_original[i * 28: (i + 1) * 28, j * 28 : (j + 1) * 28] = batch_x[j].reshape([28, 28])

        for j in range(n):
            canvas_reconstruct[i * 28 : (i + 1) * 28, j * 28 : (j + 1) * 28] = reconstruct_x[j].reshape([28, 28])

    imsave("original.png", canvas_original)
    imsave("reconstruction.png", canvas_reconstruct)
