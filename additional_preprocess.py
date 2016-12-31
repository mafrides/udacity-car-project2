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

# create jittered duplicates (for training set only)

# returns img copy randomly jittered
def jitter(source):
    # rotate
    # add padding and copy img into center
    border = 8
    new_dim = 32 + 2*border
    frame = cv2.copyMakeBorder(np.zeros((new_dim, new_dim)), border, border, border, border, cv2.BORDER_CONSTANT, value=[127, 127, 127])

    for i in range(border, border + 32):
        for j in range(border, border + 32):
            frame[i][j] = source[i - border][j - border]

    rotation_deg = random.uniform(-15, 15)
    height, width = frame.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), rotation_deg, 1.0)
    frame = cv2.warpAffine(frame, rotation_matrix, (width, height))

    # extract from center
    img = frame[border:border + 32, border:border + 32]

    # translate (fill emptied row/column with nearest row or original image)
    translation_dim1 = random.randint(-2, 2)
    dim1_order = list(range(32))
    if translation_dim1 > 0:
        dim1_order = [0] * translation_dim1 + dim1_order[:-translation_dim1]
    elif translation_dim1 < 0:
        dim1_order = dim1_order[-translation_dim1:] + [dim1_order[-1]] * -translation_dim1

    translation_dim2 = random.randint(-2, 2)
    dim2_order = list(range(32))
    if translation_dim2 > 0:
        dim2_order = [0] * translation_dim2 + dim2_order[:-translation_dim2]
    elif translation_dim2 < 0:
        dim2_order = dim2_order[-translation_dim2:] + [dim2_order[-1]] * -translation_dim2

    img = np.array([[img[i][j] for j in dim2_order] for i in dim1_order])

    # scale
    scale = random.uniform(0.9, 1.1)
    # pixels, rounded to even pixel value to assist cropping
    new_size = round(scale * 32 / 2) * 2
    scaled_img = cv2.resize(img, (new_size, new_size))

    margin = abs(new_size - 32) / 2
    index_max = 31 - margin
    if new_size > 32:
        img = scaled_img[margin:(margin + 32), margin:(margin + 32)]
    elif new_size < 32:
        # fill center of old image with new image
        for i in range(len(img)):
            for j in range(len(img[0])):
                if i >= margin and i <= index_max and j >= margin and j <= index_max:
                    img[i][j] = scaled_img[i - margin][j - margin]

    return img

# 5 jittered versions + original img
n_train = len(y_train)

X_train = np.resize(X_train, (n_train * 6, 32, 32))
y_train = np.resize(y_train, n_train * 6)

for i in range(n_train):
    if i % 1000 == 0: print('processing image', i)
    for j in range(1, 6):
        new_index = j * n_train + i
        X_train[new_index] = jitter(X_train[i])
        y_train[new_index] = y_train[i]

# multiply examples by 6 (original + 5 jittered)
print(len(X_train))