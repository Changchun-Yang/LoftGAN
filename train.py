#!/usr/bin/env python3

import sys
import pickle

import matplotlib.pyplot as plt
import numpy as np

from keras.callbacks import TensorBoard
from keras.optimizers import RMSprop, Adam

from LoftGAN.models import create_models, build_graph
from LoftGAN.training import fit_models
from LoftGAN.data import encoder_loader, decoder_loader, discriminator_loader, NUM_SAMPLES, mnist_loader, tm_data_loader
from LoftGAN.callbacks import DecoderSnapshot, ModelsCheckpoint


def set_trainable(model, trainable):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable


def main():
    encoder, decoder, discriminator = create_models()
    encoder_train, decoder_train, discriminator_train, enc, loftgan, loftgan_test = build_graph(encoder, decoder, discriminator)

    batch_size = 16
    rmsprop = RMSprop(lr=0.0003)

    train_batches = int(NUM_SAMPLES / batch_size)

    set_trainable(encoder, False)
    set_trainable(decoder, False)
    discriminator_train.compile(rmsprop, ['binary_crossentropy'] * 3, ['acc'] * 3)
    discriminator_train.summary()

    set_trainable(discriminator, False)
    set_trainable(decoder, True)
    decoder_train.compile(rmsprop, ['binary_crossentropy'] * 2, ['acc'] * 2)
    decoder_train.summary()

    phase_opt = Adam(lr=0.0005, beta_1=0.9, beta_2=0.99, decay=1e-4)

    set_trainable(decoder, False)
    set_trainable(encoder, True)
    encoder_train.compile(phase_opt)
    encoder_train.summary()
    loftgan.summary()

    set_trainable(loftgan, True)

    epochs = 30

    phase_loss = []
    D_loss = []
    G_loss = []

    for epoch in range(epochs):
        phase_epoch_loss = []
        D_epoch_loss = []
        G_epoch_loss = []
        for batch_i, (x, z_p) in enumerate(tm_data_loader(batch_size)):
            dis_loader = discriminator_loader(x, z_p)
            dec_loader = decoder_loader(x, z_p)
            enc_loader = encoder_loader(x, z_p)

            # ---------------------
            #  Train Discriminator
            # ---------------------

            dis_loss = discriminator_train.train_on_batch(dis_loader[0], dis_loader[1])

            # ---------------------
            #  Train Decoder
            # ---------------------

            dec_loss = decoder_train.train_on_batch(dec_loader[0], dec_loader[1])

            # ---------------------
            #  Train Encoder
            # ---------------------

            enc_loss = encoder_train.train_on_batch(enc_loader[0], enc_loader[1])

            # # Plot the progress
            # print("[Epoch %d/%d] [Batch %d/%d] [phase loss %.5f, acc %.5f] [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (
            # epoch, epochs, batch_i, train_batches, dis_loss[0], gp_loss[1], d_loss[0], 100 * d_loss[1],
            # g_loss))
            phase_epoch_loss.append(enc_loss[0])
            D_epoch_loss.append(dec_loss[0])
            G_epoch_loss.append(dis_loss[0])
        phase_loss.append(np.mean(phase_epoch_loss))
        D_loss.append(np.mean(D_epoch_loss))
        G_loss.append(np.mean(G_epoch_loss))
        print("[Epoch %d/%d] [phase loss %.5f] [D loss: %f] [G loss: %f]" % (epoch, epochs, phase_loss[epoch], D_loss[epoch], G_loss[epoch]))

    # loftgan.save('loftgan.h5')
    # enc.save('enc.h5')
    # loftgan_test.save('loftgan_test.h5')
    # encoder_train.save('encoder_train.h5')
    # decoder_train.save('decoder_train.h5')
    # discriminator_train.save('discriminator_train.h5')
    encoder.save('enc_2.h5')
    plt.plot(phase_loss)
    plt.title('phase loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()

    plt.plot(D_loss)
    plt.title('phase loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()

    plt.plot(G_loss)
    plt.title('phase loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.show()


if __name__ == '__main__':
    main()
