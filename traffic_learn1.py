# Load pickled data
import pickle
import numpy as np
import cv2
import random

# TODO: fill this in based on where you saved the training and testing data
training_file = 'train.p'
testing_file = 'test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

# ### Generate data additional (if you want to!)
# n_train = len(X_train)
# n_test = len(X_test)

# # add some random images that are not signs
# avg_class_count_train = int(round(n_train / n_classes))
# avg_class_count_test = int(round(n_test / n_classes))
# Not_A_Sign = 43

# def random_img():
#     return np.random.randint(0, 256, (32, 32, 3))

# X_train = np.resize(X_train, (n_train + avg_class_count_train, 32, 32, 3))
# X_test = np.resize(X_test, (n_test + avg_class_count_test, 32, 32, 3))
# y_train = np.resize(y_train, (n_train + avg_class_count_train))
# y_test = np.resize(y_test, (n_test + avg_class_count_test))

# for i in range(n_train, n_train + avg_class_count_train):
#     X_train[i] = random_img()
#     y_train[i] = Not_A_Sign

# for i in range(n_test, n_test + avg_class_count_test):
#     X_test[i] = random_img()
#     y_test[i] = Not_A_Sign

# # now there are more training examples
# n_train = len(X_train)
# n_test = len(X_test)
# print('done')

### Preprocess the data here.
### Feel free to use as many code cells as needed.
def single_channel_L_HLS(data):
    result = np.empty([len(data), 32, 32])
    for i in range(len(data)):
        result[i] = cv2.cvtColor(data[i], cv2.COLOR_RGB2HLS)[:,:,1]
    return result

# convert RGB to L of HLS
# data was n*32*32*3, now n*32*32
X_train = single_channel_L_HLS(X_train)
X_test = single_channel_L_HLS(X_test)
print('done')

# create jittered duplicates (for training set only)

# # returns img copy randomly jittered
# def jitter(source):
#     # rotate
#     # add padding and copy img into center
#     border = 15
#     new_dim = 32 + 2*border
#     frame = cv2.copyMakeBorder(np.zeros((new_dim, new_dim)), border, border, border, border, cv2.BORDER_CONSTANT, value=[127, 127, 127])

#     for i in range(border, border + 32):
#         for j in range(border, border + 32):
#             frame[i][j] = source[i - border][j - border]

#     rotation_deg = random.uniform(-15, 15)
#     height, width = frame.shape[:2]
#     rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), rotation_deg, 1.0)
#     frame = cv2.warpAffine(frame, rotation_matrix, (width, height))

#     # extract from center
#     img = frame[border:border + 32, border:border + 32]

#     # translate (fill emptied row/column with nearest row or original image)
#     translation_dim1 = random.randint(-2, 2)
#     dim1_order = list(range(32))
#     if translation_dim1 > 0:
#         dim1_order = [0] * translation_dim1 + dim1_order[:-translation_dim1]
#     elif translation_dim1 < 0:
#         dim1_order = dim1_order[-translation_dim1:] + [dim1_order[-1]] * -translation_dim1

#     translation_dim2 = random.randint(-2, 2)
#     dim2_order = list(range(32))
#     if translation_dim2 > 0:
#         dim2_order = [0] * translation_dim2 + dim2_order[:-translation_dim2]
#     elif translation_dim2 < 0:
#         dim2_order = dim2_order[-translation_dim2:] + [dim2_order[-1]] * -translation_dim2

#     img = np.array([[img[i][j] for j in dim2_order] for i in dim1_order])

#     # scale
#     scale = random.uniform(0.9, 1.1)
#     # pixels, rounded to even pixel value to assist cropping
#     new_size = round(scale * 32 / 2) * 2
#     scaled_img = cv2.resize(img, (new_size, new_size))

#     margin = abs(new_size - 32) / 2
#     index_max = 31 - margin
#     if new_size > 32:
#         img = scaled_img[margin:(margin + 32), margin:(margin + 32)]
#     elif new_size < 32:
#         # fill center of old image with new image
#         for i in range(len(img)):
#             for j in range(len(img[0])):
#                 if i >= margin and i <= index_max and j >= margin and j <= index_max:
#                     img[i][j] = scaled_img[i - margin][j - margin]

#     return img

# # 5 jittered versions + original img
# X_train = np.resize(X_train, (n_train * 6, 32, 32))
# y_train = np.resize(y_train, n_train * 6)

# for i in range(n_train):
#     if i % 1000 == 0: print('processing image', i)
#     for j in range(1, 6):
#         new_index = j * n_train + i
#         X_train[new_index] = jitter(X_train[i])
#         y_train[new_index] = y_train[i]

# # multiply examples by 6 (original + 5 jittered)
# n_train = len(X_train)
# print(n_train)

# # sanity check for jittered imgs
# import matplotlib.pyplot as plot

# def show_img_plot_gray(name, img, figure_num):
#     plot.figure(figure_num)
#     plot.imshow(img, cmap='gray')
#     plot.title(name)
#     plot.show()

# for i in range(1000, n_train, int(n_train / 6)):
#     show_img_plot_gray('Class 0, example ' + str(i), X_train[i], i)

# def local_normalize(data):
#     for i in range(len(data)):
#         data[i] = cv2.blur(data[i], (3, 3))
#         # data[i] = cv2.bilateralFilter(data[i], 5, 10, 10)

# local_normalize(X_train)
# local_normalize(X_test)
# print('done')

# # img sample with blur
# import matplotlib.pyplot as plot

# def show_img_plot_gray(name, img, figure_num):
#     plot.figure(figure_num)
#     plot.imshow(img, cmap='gray')
#     plot.title(name)
#     plot.show()

# for i in range(1000, n_train, int(n_train / 6)):
#     show_img_plot_gray('Class 0, example ' + str(i) + ' with blur', X_train[i], i)

def global_normalize(data):
    for i in range(len(data)):
        data[i] = cv2.normalize(data[i], data[i], alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

# normalize pixels to [0, 1]
global_normalize(X_train)
global_normalize(X_test)
print('done')

# # pickle preprocessed data
# import pickle

# with open('x_train_normalized.p', mode='wb') as f:
#     pickle.dump(X_train, f)
# with open('y_train_normalized.p', mode='wb') as f:
#     pickle.dump(y_train, f)
# with open('x_test_normalized.p', mode='wb') as f:
#     pickle.dump(X_test, f)
# with open('y_test_normalized.p', mode='wb') as f:
#     pickle.dump(y_test, f)

# print('preprocessed data saved')

# # unpickle preprocessed data
# import pickle
# import numpy as np

# with open('x_train_normalized.p', mode='rb') as f:
#     X_train = pickle.load(f)
# with open('y_train_normalized.p', mode='rb') as f:
#     y_train = pickle.load(f)
# with open('x_test_normalized.p', mode='rb') as f:
#     X_test = pickle.load(f)
# with open('y_test_normalized.p', mode='rb') as f:
#     y_test = pickle.load(f)

# print('preprocessed data loaded')


n_classes = 43
### and split the data into training/validation/testing sets here.
### Feel free to use as many code cells as needed.
import random

train_set_indexes_by_class = [[] for i in range(n_classes)]

for i in range(len(y_train)):
    class_number = y_train[i]
    train_set_indexes_by_class[class_number].append(i)

for i in range(n_classes):
    random.shuffle(train_set_indexes_by_class[i])

print('done')

# use last 10% of each bucket for validation
# distribute requests as evenly as possible
import math

batch_counter = [None] * n_classes
for i in range(n_classes):
    batch_counter[i] = random.randint(0, math.floor(len(train_set_indexes_by_class[i]) * .9) - 1)

def get_next_batch(n_examples, counter=batch_counter):
    count_per_class = [None] * n_classes
    number_per_class = math.floor(n_examples / n_classes)
    extras = n_examples % n_classes
    for i in range(n_classes):
        if (i < extras):
            count_per_class[i] = number_per_class + 1
        else:
            count_per_class[i] = number_per_class

    result = np.empty([n_examples, 32, 32])
    result_index = 0

    result_y = np.empty([n_examples])
    result_y_index = 0

    # get examples for each class
    for i in range(n_classes):
        class_train_set_indexes = train_set_indexes_by_class[i]
        count = count_per_class[i]
        counter = batch_counter[i]
        max_index = math.floor(len(class_train_set_indexes) * .9)

        if (counter + count <= max_index):
            for j in range(count):
                result[result_index + j] = X_train[class_train_set_indexes[counter + j]]

            batch_counter[i] += count
            result_index += count
        else:
            from_end_count = max_index - counter + 1
            from_beginning_count = count - from_end_count

            for k in range(from_end_count):
                result[result_index + k] = X_train[class_train_set_indexes[counter + k]]

            result_index += from_end_count

            for m in range(from_beginning_count):
                result[result_index + m] = X_train[class_train_set_indexes[m]]

            result_index += from_beginning_count
            batch_counter[i] = from_beginning_count

        for n in range(count):
            result_y[result_y_index + n] = i

        result_y_index += count

    return result, result_y

def get_validation_batch():
    counter = 0
    for i in range(n_classes):
        class_train_set_indexes = train_set_indexes_by_class[i]
        counter += len(class_train_set_indexes) - math.floor(len(class_train_set_indexes) * .9) - 1

    print('validation batch size', counter)

    result_x = np.empty([counter, 32, 32])
    result_y = np.empty([counter])

    result_index = 0

    for i in range(n_classes):
        class_train_set_indexes = train_set_indexes_by_class[i]
        for j in range(math.floor(len(class_train_set_indexes) * .9) + 1, len(class_train_set_indexes)):
            result_x[result_index] = X_train[class_train_set_indexes[j]]
            result_y[result_index] = i
            result_index += 1

    return result_x, result_y

print('done')

validation_x, validation_y = get_validation_batch()
print('constructed validation batch', len(validation_x), 'examples')

### Define your architecture here.
### Feel free to use as many code cells as needed.
import tensorflow as tf

def conv_net(x):
    x = tf.expand_dims(x, 3)

    x_flat = tf.reshape(x, [-1, 32*32])

    conv1_depth = 16

    W1 = tf.Variable(tf.truncated_normal([5, 5, 1, conv1_depth]))
    b1 = tf.Variable(tf.zeros(conv1_depth))
    conv1 = tf.nn.conv2d(x, W1, strides=[1, 2, 2, 1], padding='VALID')
    conv1 = tf.nn.bias_add(conv1, b1)
    conv1 = tf.nn.relu(conv1)

    conv1_flat = tf.reshape(conv1, [-1, 14*14*conv1_depth])

    fc1_width = 20 * n_classes

    fc1 = tf.concat(1, [x_flat, conv1_flat])
    fc1_W = tf.Variable(tf.truncated_normal(shape=(32*32 + 14*14*conv1_depth, fc1_width)))
    fc1_b = tf.Variable(tf.zeros(fc1_width))
    fc1 = tf.matmul(fc1, fc1_W) + fc1_b
    fc1 = tf.nn.relu(fc1)

    fc2_W = tf.Variable(tf.truncated_normal(shape=(fc1_width, n_classes)))
    fc2_b = tf.Variable(tf.zeros(n_classes))
    result = tf.matmul(fc1, fc2_W) + fc2_b

    return result

x = tf.placeholder("float", [None, 32, 32])
y = tf.placeholder("float", [None])

y_one_hot = tf.one_hot(tf.cast(y, tf.int32), n_classes)

logits = conv_net(x)

learning_rate = tf.placeholder(tf.float32, shape=[])
momentum = tf.placeholder(tf.float32, shape=[])
batch_size = 4 * n_classes
training_epochs = 100
n_classes = n_classes

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, y_one_hot))
optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=momentum, use_nesterov=True)\
    .minimize(cost)

# validation
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_one_hot, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

init = tf.initialize_all_variables()
print('done')

### Train your model here.
### Feel free to use as many code cells as needed.
print(len(y_train), 'training examples')
print('batch size', batch_size)
print('epochs', training_epochs)
print(int(len(y_train) / batch_size), 'batches per epoch')

saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    # Training cycle
    for epoch in range(training_epochs):
        total_batch = int(len(y_train) / batch_size)
        # Loop over all batches
        epoch_learning_rate = .002 / math.ceil((epoch + 1) / 10)
        epoch_momentum = 0.9
        # - math.floor((epoch + 1) / 15) * 0.05
        for i in range(total_batch):
            batch_x, batch_y = get_next_batch(batch_size)
            # Run optimization op (backprop) and cost op (to get loss value)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y, learning_rate: epoch_learning_rate, momentum: epoch_momentum })
        # Display logs per epoch step
        c = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
        print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c))
        if epoch > 0 and epoch % 5 == 0:
          print("Accuracy:", accuracy.eval({x: validation_x, y: validation_y}))
    print("Optimization Finished!")

    # Save the variables to disk.
    save_path = saver.save(sess, "/Users/afrides/udacity/car/project2/model_1.ckpt")
    print("Model saved in file: %s" % save_path)
