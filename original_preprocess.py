# Load pickled data
import pickle
import numpy as np
import cv2

# TODO: fill this in based on where you saved the training and testing data
training_file = 'train.p'
testing_file = 'test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

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
print('done converting to single channel')

def global_normalize(data):
    for i in range(len(data)):
        data[i] = cv2.normalize(data[i], data[i], alpha=-0.5, beta=0.5, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

# normalize pixels to [0, 1]
global_normalize(X_train)
global_normalize(X_test)
print('done normalizing')

with open('x_train_normalized_1.p', mode='wb') as f:
    pickle.dump(X_train, f)
with open('y_train_normalized_1.p', mode='wb') as f:
    pickle.dump(y_train, f)
with open('x_test_normalized_1.p', mode='wb') as f:
    pickle.dump(X_test, f)
with open('y_test_normalized_1.p', mode='wb') as f:
    pickle.dump(y_test, f)

print('preprocessed data without additional data saved')