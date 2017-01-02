# unpickle preprocessed data without additional data
import pickle

with open('x_train_normalized_1.p', mode='rb') as f:
    X_train = pickle.load(f)
with open('y_train_normalized_1.p', mode='rb') as f:
    y_train = pickle.load(f)
with open('x_test_normalized_1.p', mode='rb') as f:
    X_test = pickle.load(f)
with open('y_test_normalized_1.p', mode='rb') as f:
    y_test = pickle.load(f)

print('preprocessed data without additional data loaded')

### and split the data into training/validation/testing sets here.
### Feel free to use as many code cells as needed.
from sklearn.model_selection import train_test_split
import numpy as np

# TODO: Use `train_test_split` here.
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

print('done splitting training and validation sets')
print(len(X_train), len(X_val), len(y_train), len(y_val))

### Define your architecture here.
### Feel free to use as many code cells as needed.
import tensorflow as tf

def conv_net(x):
    x = tf.expand_dims(x, 3)

    # 30x30x32
    W1 = tf.Variable(tf.truncated_normal([3, 3, 1, 32]))
    b1 = tf.Variable(tf.zeros(32))
    conv1 = tf.nn.conv2d(x, W1, strides=[1, 1, 1, 1], padding='VALID')
    conv1 = tf.nn.bias_add(conv1, b1)
    conv1 = tf.nn.relu(conv1)

    # 28x28x32
    W2 = tf.Variable(tf.truncated_normal([3, 3, 32, 32]))
    b2 = tf.Variable(tf.zeros(32))
    conv2 = tf.nn.conv2d(conv1, W2, strides=[1, 1, 1, 1], padding='VALID')
    conv2 = tf.nn.bias_add(conv2, b2)
    conv2 = tf.nn.relu(conv2)

    # 14x14x32
    pool = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')

    pool = tf.nn.dropout(pool, 0.25)

    fc1 = tf.reshape(pool, [-1, 14*14*32])
    fc1_W = tf.Variable(tf.truncated_normal(shape=(14*14*32, 128)))
    fc1_b = tf.Variable(tf.zeros(128))
    fc1 = tf.matmul(fc1, fc1_W) + fc1_b
    fc1 = tf.nn.relu(fc1)

    fc1 = tf.nn.dropout(fc1, 0.5)

    fc2_W = tf.Variable(tf.truncated_normal(shape=(128, 43)))
    fc2_b = tf.Variable(tf.zeros(43))
    result = tf.matmul(fc1, fc2_W) + fc2_b

    return result

x = tf.placeholder("float", [None, 32, 32])
y = tf.placeholder("float", [None])

y_one_hot = tf.one_hot(tf.cast(y, tf.int32), 43)

logits = conv_net(x)

learning_rate = 0.001

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y_one_hot))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# validation
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

init = tf.initialize_all_variables()

### Train your model here.
### Feel free to use as many code cells as needed.
saver = tf.train.Saver()

batch_size = 128
training_epochs = 20

print(len(y_train), 'training examples')
print('learning rate', learning_rate)
print('batch size', batch_size)
print('epochs', training_epochs)

with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        # Loop over all batches
        i = 0
        while i < len(y_train):
            batch_x = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            i = i + batch_size
        # Display logs per epoch step
        c = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
        print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))

    print("Accuracy:", accuracy.eval({x: X_val, y: y_val}))
    print("Optimization Finished!")

    # Save the variables to disk.
    save_path = saver.save(sess, "/Users/afrides/udacity/car/project2/model_1.ckpt")
    print("Model saved in file: %s" % save_path)
