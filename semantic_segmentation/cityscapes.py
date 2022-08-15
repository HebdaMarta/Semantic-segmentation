import h5py
import numpy as np
import random
import cv2
from keras.preprocessing.image import ImageDataGenerator
from __future__ import print_function
import os
import argparse
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model
from keras.layers import Input
from keras.layers.core import Lambda, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Add
from keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf


def pre_processing(img):
    rand_s = random.uniform(0.9, 1.1)
    rand_v = random.uniform(0.9, 1.1)

    img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    tmp = np.ones_like(img[:, :, 1]) * 255
    img[:, :, 1] = np.where(img[:, :, 1] * rand_s > 255, tmp, img[:, :, 1] * rand_s)
    img[:, :, 2] = np.where(img[:, :, 2] * rand_v > 255, tmp, img[:, :, 2] * rand_v)

    img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

    return img / 127.5 - 1


# ImageDataGenerator
def get_data_gen_args(mode):
    if mode == 'train' or mode == 'val':
        x_data_gen_args = dict(preprocessing_function=pre_processing,
                               shear_range=0.1,
                               zoom_range=0.1,
                               rotation_range=10,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               fill_mode='constant',
                               horizontal_flip=True)

        y_data_gen_args = dict(shear_range=0.1,
                               zoom_range=0.1,
                               rotation_range=10,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               fill_mode='constant',
                               horizontal_flip=True)

    else:
        print("Data_generator function should get mode arg 'train' or 'val'.")
        return -1

    return x_data_gen_args, y_data_gen_args


# One hot encoding y_img.
def get_result_map(b_size, y_img):
    y_img = np.squeeze(y_img, axis=3)
    result_map = np.zeros((b_size, 256, 512, 4))

    person = (y_img == 24)
    car = (y_img == 26)
    road = (y_img == 7)

    background = np.logical_not(person + car + road)

    result_map[:, :, :, 0] = np.where(background, 1, 0)
    result_map[:, :, :, 1] = np.where(person, 1, 0)
    result_map[:, :, :, 2] = np.where(car, 1, 0)
    result_map[:, :, :, 3] = np.where(road, 1, 0)

    return result_map


# Data generator
def data_generator(d_path, b_size, mode):
    data = h5py.File(d_path, 'r')
    x_imgs = data.get('/' + mode + '/x')
    y_imgs = data.get('/' + mode + '/y')

    x_data_gen_args, y_data_gen_args = get_data_gen_args(mode)
    x_data_gen = ImageDataGenerator(**x_data_gen_args)
    y_data_gen = ImageDataGenerator(**y_data_gen_args)

    d_size = x_imgs.shape[0]
    shuffled_idx = list(range(d_size))

    x = []
    y = []
    while True:
        random.shuffle(shuffled_idx)
        for i in range(d_size):
            idx = shuffled_idx[i]

            x.append(x_imgs[idx].reshape((256, 512, 3)))
            y.append(y_imgs[idx].reshape((256, 512, 1)))

            if len(x) == b_size:
                _ = np.zeros(b_size)
                seed = random.randrange(1, 1000)

                x_tmp_gen = x_data_gen.flow(np.array(x), _,
                                            batch_size=b_size,
                                            seed=seed)
                y_tmp_gen = y_data_gen.flow(np.array(y), _,
                                            batch_size=b_size,
                                            seed=seed)

                # Finally, yield x, y data.
                x_result, _ = next(x_tmp_gen)
                y_result, _ = next(y_tmp_gen)

                yield x_result, get_result_map(b_size, y_result)

                x.clear()
                y.clear()

dir_path = os.path.dirname('/content/drive/MyDrive/Colab Notebooks/')
img_folder_path = '/content/drive/MyDrive/BAZY DANYCH/ULICE W MIASTACH - CITYSCAPES/raw_imgs'
gt_folder_path = '/content/drive/MyDrive/BAZY DANYCH/ULICE W MIASTACH - CITYSCAPES/segment_imgs'

# check only three labels = ['background', 'person', 'car', 'road']

def get_data(mode):
    if mode == 'train/zurich' or mode == 'val/munster':
        x_paths = []
        y_paths = []
        tmp_img_folder_path = os.path.join(img_folder_path, mode)
        tmp_gt_folder_path = os.path.join(gt_folder_path, mode)

        for (path, dirname, files) in os.walk(tmp_img_folder_path):
            for filename in files:
                x_paths.append(os.path.join(path, filename))

        # Find ground_truth file paths with x_paths.
        idx = len(tmp_img_folder_path)
        for x_path in x_paths:
            y_paths.append(tmp_gt_folder_path + x_path[idx:-15] + 'gtFine_labelIds.png')

        return x_paths, y_paths
    else:
        print("Please call get_data function with arg 'train', 'val'.")


# Make h5 group and write data
def write_data(h5py_file, mode, x_paths, y_paths):
    num_data = len(x_paths)

    # h5py special data type for image.
    uint8_dt = h5py.special_dtype(vlen=np.dtype('uint8'))

    # Make group and data set.
    group = h5py_file.create_group(mode)
    x_dset = group.create_dataset('x', shape=(num_data, ), dtype=uint8_dt)
    y_dset = group.create_dataset('y', shape=(num_data, ), dtype=uint8_dt)

    for i in range(num_data):
        # Read image and resize
        x_img = cv2.imread(x_paths[i])
        x_img = cv2.resize(x_img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
        x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2RGB)

        y_img = cv2.imread(y_paths[i])
        y_img = cv2.resize(y_img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_NEAREST)
        y_img = y_img[:, :, 0]

        x_dset[i] = x_img.flatten()
        y_dset[i] = y_img.flatten()


# Make h5 file
def make_h5py():
    x_train_paths, y_train_paths = get_data('train/zurich')
    x_val_paths, y_val_paths = get_data('val/munster')

    # Make h5py file with write option.
    h5py_file = h5py.File(os.path.join(dir_path, 'data.h5'), 'w')

    # Write data
    print('Parsing train datas...')
    write_data(h5py_file, 'train', x_train_paths, y_train_paths)
    print('Finish.')

    print('Parsing val datas...')
    write_data(h5py_file, 'val', x_val_paths, y_val_paths)
    print('Finish.')

make_h5py()

def dice_coef(y_true, y_pred):
    return (2. * K.sum(y_true * y_pred) + 1.) / (K.sum(y_true) + K.sum(y_pred) + 1.)


def fcn_8s(num_classes, input_shape, lr_init, lr_decay, vgg_weight_path=None):
    img_input = Input(input_shape)

    # Block 1
    x = Conv2D(64, (3, 3), padding='same', name='block1_conv1')(img_input)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, (3, 3), padding='same', name='block1_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling2D()(x)

    # Block 2
    x = Conv2D(128, (3, 3), padding='same', name='block2_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128, (3, 3), padding='same', name='block2_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling2D()(x)

    # Block 3
    x = Conv2D(256, (3, 3), padding='same', name='block3_conv1')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), padding='same', name='block3_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, (3, 3), padding='same', name='block3_conv3')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    block_3_out = MaxPooling2D()(x)

    # Block 4
    x = Conv2D(512, (3, 3), padding='same', name='block4_conv1')(block_3_out)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same', name='block4_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same', name='block4_conv3')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    block_4_out = MaxPooling2D()(x)

    # Block 5
    x = Conv2D(512, (3, 3), padding='same', name='block5_conv1')(block_4_out)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same', name='block5_conv2')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(512, (3, 3), padding='same', name='block5_conv3')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = MaxPooling2D()(x)

    # Load pretrained weights.
    if vgg_weight_path is not None:
        vgg16 = Model(img_input, x)
        vgg16.load_weights(vgg_weight_path, by_name=True)

    x = Conv2D(4096, (7, 7), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(4096, (1, 1), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(num_classes, (1, 1), strides=(1, 1), activation='linear')(x)
    x = BatchNormalization()(x)

    block_3_out = Conv2D(num_classes, (1, 1), strides=(1, 1), activation='linear')(block_3_out)
    block_3_out = BatchNormalization()(block_3_out)

    block_4_out = Conv2D(num_classes, (1, 1), strides=(1, 1), activation='linear')(block_4_out)
    block_4_out = BatchNormalization()(block_4_out)

    x = Lambda(lambda x: tf.image.resize(x, (x.shape[1] * 2, x.shape[2] * 2)))(x)
    x = Add()([x, block_4_out])
    x = Activation('relu')(x)

    x = Lambda(lambda x: tf.image.resize(x, (x.shape[1] * 2, x.shape[2] * 2)))(x)
    x = Add()([x, block_3_out])
    x = Activation('relu')(x)

    x = Lambda(lambda x: tf.image.resize(x, (x.shape[1] * 8, x.shape[2] * 8)))(x)

    x = Activation('softmax')(x)

    model = Model(img_input, x)
    model.compile(optimizer=Adam(lr=lr_init, decay=lr_decay),
                  loss='categorical_crossentropy',
                  metrics=[dice_coef])

    return model

dir_path = os.path.dirname('/content/drive/MyDrive/Colab Notebooks/')

TRAIN_BATCH = 4
VAL_BATCH = 1
lr_init = 1e-4
lr_decay = 5e-4

vgg_path = '/content/drive/MyDrive/SAMOCHODY/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5'

# Use only 3 classes
labels = ['background', 'person', 'car', 'road']

# Model
model = fcn_8s(input_shape=(256, 512, 3), num_classes=len(labels), lr_init=lr_init, lr_decay=lr_decay, vgg_weight_path=vgg_path)

# training
history = model.fit_generator(data_generator('/content/drive/MyDrive/Colab Notebooks/data.h5', TRAIN_BATCH, 'train'),
                              steps_per_epoch=3475 // TRAIN_BATCH,
                              validation_data=data_generator('/content/drive/MyDrive/Colab Notebooks/data.h5', VAL_BATCH, 'val'),
                              validation_steps=500 // VAL_BATCH,
                              epochs=20,
                              verbose=1)
# Charts
plt.title("loss")
plt.plot(history.history["loss"], color="r", label="train")
plt.plot(history.history["val_loss"], color="b", label="val")
plt.legend(loc="best")
plt.show()

plt.gcf().clear()
plt.title("dice_coef")
plt.plot(history.history["dice_coef"], color="r", label="train")
plt.plot(history.history["val_dice_coef"], color="b", label="val")
plt.legend(loc="best")

# Test

def result_map_to_img(res_map):
    img = np.zeros((256, 512, 3), dtype=np.uint8)
    res_map = np.squeeze(res_map)

    argmax_idx = np.argmax(res_map, axis=2)

    person = (argmax_idx == 1)
    car = (argmax_idx == 2)
    road = (argmax_idx == 3)


    img[:, :, 0] = np.where(person, 255, 0)
    img[:, :, 1] = np.where(car, 255, 0)
    img[:, :, 2] = np.where(road, 255, 0)

    return img


label = {0: 'background', 1:'person', 2: "car", 3: 'road'}
from google.colab.patches import cv2_imshow
img_path = '/content/drive/MyDrive/BAZY DANYCH/ULICE W MIASTACH - CITYSCAPES/raw_imgs/test/bielefeld/bielefeld_000000_030958_leftImg8bit.png' 


x_img = cv2.imread(img_path)
x_img = cv2.resize(x_img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_CUBIC)
x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2RGB)

cv2_imshow(x_img)
x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2RGB)
x_img = x_img / 127.5 - 1
x_img = np.expand_dims(x_img, 0)

x_img = x_img.reshape(-1, 256, 512, 3)
pred = model.predict(x_img)
res = result_map_to_img(pred[0])

res = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)

cv2_imshow(res)
cv2.waitKey(0)