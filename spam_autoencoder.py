import tensorflow as tf
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import homogeneity_score
from sklearn.cluster import KMeans

file_path = 'tmp/spambase_autoencoder/'
file_output = "encoded_spam_X_2.npy"

# parameters
learning_rate = 0.01
num_steps = 20000
batch_size = 256
display_step = 1000

# Network parameters
num_hidden = 10
num_input = 54

X = tf.placeholder(tf.float32, [None, num_input])

weights = {
    "encoder_w1": tf.Variable(tf.random_normal([num_input, num_hidden])),
    "decoder_w1": tf.Variable(tf.random_normal([num_hidden, num_input])),
    "encoder_b1": tf.Variable(tf.random_normal([num_hidden])),
    "decoder_b1": tf.Variable(tf.random_normal([num_input])),
}

def encoder(X):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(X, weights["encoder_w1"]), weights["encoder_b1"]))
    return layer_1


def decoder(encoded_X):
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(encoded_X, weights["decoder_w1"]), weights["decoder_b1"]))
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

with tf.Session() as sess:
    data = pd.read_csv('data/spambase/spambase.data', dtype='float', names=[i for i in range(56)])
    data = data.values
    spam_data_X = data[:, :54]
    spam_data_y = data[:, 55]
    train_X, test_X, train_y, test_y = train_test_split(spam_data_X, spam_data_y)

    dataset = tf.data.Dataset.from_tensor_slices(train_X)
    dataset = dataset.repeat().batch(batch_size)
    iterator = dataset.make_one_shot_iterator()
    next_item = iterator.get_next()

    sess.run(init)

    for i in range(num_steps):
        batch_x = sess.run(next_item)
        _, loss_ = sess.run([train_op, loss], feed_dict={X: batch_x})
        if i % display_step == 0:
            print("Step: ", i, " Loss: " + "{:.3f}".format(loss_))

    train_encoded = sess.run(encode, feed_dict={X: train_X})
    test_encoded = sess.run(encode, feed_dict={X: test_X})

    clf = LogisticRegression()
    clf.fit(train_encoded, train_y)
    print("Train Accuracy ", clf.score(train_encoded, train_y))
    print("Train Accuracy ", clf.score(test_encoded, test_y))

    clu = KMeans(n_clusters=2)
    clu.fit(train_encoded)
    print("Train purity score", homogeneity_score(train_y, clu.predict(train_encoded)))
    print("Test purity score", homogeneity_score(test_y, clu.predict(test_encoded)))

    #np.save(file_output, train_encoded)
