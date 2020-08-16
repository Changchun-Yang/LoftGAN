#!/usr/bin/env python3

import os.path
import glob
from multiprocessing.pool import ThreadPool as Pool

import numpy as np
from PIL import Image
import scipy.io as scio


NUM_SAMPLES = 12288

proj_root = os.path.split(os.path.dirname(__file__))[0]
images_path = os.path.join(proj_root, 'img_align_celeba_png', '*.png')

data_path = '../gen_data/pair_data.mat'

disc_patch = (4, 4, 1)


def _load_image(f):
    im = Image.open(f) \
              .crop((0, 20, 178, 198)) \
              .resize((64, 64), Image.BICUBIC)
    return np.asarray(im)


def celeba_loader(batch_size, normalize=True, num_child=4, seed=0, workers=8):
    rng = np.random.RandomState(seed)
    images = glob.glob(images_path)

    with Pool(workers) as p:
        while True:
            rng.shuffle(images)
            for s in range(0, len(images), batch_size):
                e = s + batch_size
                batch_names = images[s:e]
                batch_images = p.map(_load_image, batch_names)
                batch_images = np.stack(batch_images)

                if normalize:
                    batch_images = batch_images / 127.5 - 1.
                    # To be sure
                    batch_images = np.clip(batch_images, -1., 1.)

                # Yield the same batch num_child times since the images will be consumed
                # by num_child different child generators
                for i in range(num_child):
                    yield batch_images


def tm_data_loader(batch_size):

    train_labels = scio.loadmat(data_path)['phase_matrix'][0:NUM_SAMPLES]
    train_images = scio.loadmat(data_path)['speckle_matrix'][0:NUM_SAMPLES].reshape(NUM_SAMPLES, 64, 64, 1)

    nRows, nCols, nDims = train_images.shape[1:]
    train_data = train_images.reshape(train_images.shape[0], nRows, nCols, nDims)

    train_out = train_labels

    train_batches = int(NUM_SAMPLES / batch_size)

    for i in range(train_batches):
        batch_X = train_data[i*batch_size:(i+1)*batch_size]
        batch_y = train_out[i*batch_size:(i+1)*batch_size]

        yield batch_X, batch_y


def mnist_loader(batch_size, normalize=True, num_child=4, seed=0, workers=8):
    from keras.datasets import mnist
    (x_train, _), (_, _) = mnist.load_data()

    x_train_new = np.zeros((x_train.shape[0], 64, 64), dtype='int32')

    for i, img in enumerate(x_train):
        im = Image.fromarray(img).resize((64, 64), Image.BICUBIC)
        x_train_new[i] = np.asarray(im)

    x_train = x_train_new.reshape(-1, 64, 64, 1)
    del x_train_new

    if normalize:
        x_train = x_train / 127.5 - 1.
        # To be sure
        x_train = np.clip(x_train, -1., 1.)

    rng = np.random.RandomState(seed)
    while True:
        rng.shuffle(x_train)
        for s in range(0, len(x_train), batch_size):
            e = s + batch_size
            batch_images = x_train[s:e]

            # Yield the same batch num_child times since the images will be consumed
            # by num_child different child generators
            for i in range(num_child):
                yield batch_images


def discriminator_loader(x, z_p):
    batch_size = x.shape[0]

    y_real = np.ones((batch_size,) + disc_patch, dtype='float32')
    y_fake = np.zeros((batch_size,) + disc_patch, dtype='float32')

    return [x, z_p], [y_real, y_fake, y_fake]


def decoder_loader(x, z_p):

    batch_size = x.shape[0]

    y_real = np.ones((batch_size,) + disc_patch, dtype='float32')
    return [x, z_p], [y_real, y_real]


def encoder_loader(x, z_p):
    return [x, z_p], None
