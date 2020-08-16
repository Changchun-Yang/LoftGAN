#!/usr/bin/env python3

import numpy as np

from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, BatchNormalization, Dense, Conv2DTranspose, Flatten, Reshape, \
    Lambda, LeakyReLU, Activation, Dropout
from keras.regularizers import l2
from keras.losses import KLD, mse

from LoftGAN.losses import mean_gaussian_negative_log_likelihood


def create_models(n_channels=1, recon_depth=9, wdecay=1e-5, bn_mom=0.9, bn_eps=1e-6):

    image_shape = (64, 64, n_channels)
    latent_dim = (1024,)
    decode_from_shape = (8, 8, 48)
    n_decoder = np.prod(decode_from_shape)

    # Encoder
    def create_encoder():

        cnn_input = Input(shape=image_shape, name='enc_input')

        conv1 = Conv2D(16, (7, 7), strides=(3, 3), padding='valid',
                       kernel_initializer='glorot_normal',
                       input_shape=image_shape,
                       activation='tanh')(cnn_input)

        conv2 = Conv2D(32, (5, 5), strides=(2, 2), padding='same',
                       kernel_initializer='glorot_normal',
                       activation='tanh')(conv1)

        conv3 = Conv2D(48, (3, 3), strides=(1, 1), padding='valid',
                       kernel_initializer='glorot_normal',
                       activation='tanh')(conv2)

        flat1 = Flatten()(conv3)

        d2 = Dense(512, activation='tanh', kernel_initializer='glorot_normal')(flat1)
        d3 = Dropout(0.5)(d2)

        output = Dense(32 * 32, activation='sigmoid', kernel_initializer='glorot_normal', name='phase_output')(d3)

        # z_mean = Dense(latent_dim, activation='sigmoid', name='z_mean', kernel_initializer='glorot_normal')(d3)
        # z_log_var = Dense(latent_dim, activation='sigmoid', name='z_log_var', kernel_initializer='glorot_normal')(d3)

        return Model(inputs=cnn_input, outputs=output, name='encoder')

        # return Model(cnn_input, [z_mean, z_log_var], name='encoder')

    def create_decoder():
        z = Input(shape=latent_dim)

        x = Dense(512, activation='tanh', kernel_initializer='glorot_normal', input_shape=latent_dim)(z)
        x = Dense(n_decoder, activation='tanh', kernel_initializer='glorot_normal')(x)
        x = Dropout(0.5)(x)
        x = Reshape(decode_from_shape)(x)

        x = Conv2DTranspose(32, (3, 3), strides=(1, 1), padding='valid',
                       kernel_initializer='glorot_normal',
                       activation='tanh')(x)

        x = Conv2DTranspose(16, (5, 5), strides=(2, 2), padding='same',
                       kernel_initializer='glorot_normal',
                       activation='tanh')(x)

        x = Conv2DTranspose(1, (7, 7), strides=(3, 3), padding='valid',
                       kernel_initializer='glorot_normal',
                       activation='tanh', name='decoder_output')(x)

        return Model(z, x, name='decoder')

    def create_discriminator():

        d_input = Input(image_shape, name='dis_input')

        conv1 = Conv2D(16, (7, 7), strides=(3, 3), padding='valid',
                       kernel_initializer='glorot_normal',
                       input_shape=image_shape,
                       activation='tanh')(d_input)

        conv2 = Conv2D(32, (5, 5), strides=(2, 2), padding='same',
                       kernel_initializer='glorot_normal',
                       activation='tanh')(conv1)

        conv3 = Conv2D(48, (3, 3), strides=(1, 1), padding='valid',
                       kernel_initializer='glorot_normal',
                       activation='tanh', name='dis_feat_out')(conv2)

        y_feat = conv3

        conv3 = Conv2D(48, (3, 3), strides=(1, 1), padding='valid',
                       kernel_initializer='glorot_normal',
                       activation='tanh')(conv3)

        conv3 = Conv2D(48, (3, 3), strides=(1, 1), padding='valid',
                       kernel_initializer='glorot_normal',
                       activation='tanh')(conv3)

        y = Conv2D(1, kernel_size=3, strides=1, padding='same', name='dis_output')(conv3)

        return Model(d_input, [y, y_feat], name='discriminator')

    encoder = create_encoder()
    decoder = create_decoder()
    discriminator = create_discriminator()
    encoder.summary()
    decoder.summary()
    discriminator.summary()
    return encoder, decoder, discriminator


def _sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
       Instead of sampling from Q(z|X), sample eps = N(0,I)
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var, z_p = args
    # batch = K.shape(z_mean)[0]
    # dim = K.int_shape(z_mean)[1]
    # # by default, random_normal has mean=0 and std=1.0
    # epsilon = K.random_normal(shape=(batch, dim))
    epsilon = z_p
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


def build_graph(encoder, decoder, discriminator, recon_vs_gan_weight=1e-6):
    image_shape = K.int_shape(encoder.input)[1:]
    latent_shape = K.int_shape(decoder.input)[1:]

    sampler = Lambda(_sampling, output_shape=latent_shape, name='sampler')

    # Inputs
    x = Input(shape=image_shape, name='input_image')
    # z_p is sampled directly from isotropic gaussian
    z_p = Input(shape=latent_shape, name='z_p')

    # Build computational graph

    # z_mean, z_log_var = encoder(x)
    # z = sampler([z_mean, z_log_var, z_p])

    z = encoder(x)

    x_tilde = decoder(z)
    x_p = decoder(z_p)

    dis_x, dis_feat = discriminator(x)
    dis_x_tilde, dis_feat_tilde = discriminator(x_tilde)
    dis_x_p = discriminator(x_p)[0]

    # Compute losses

    # Learned similarity metric
    dis_nll_loss = mean_gaussian_negative_log_likelihood(dis_feat, dis_feat_tilde)

    # KL divergence loss
    # kl_loss = K.mean(-0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1))

    kl_loss = KLD(z_p, z)

    phase_loss = mse(z_p, z)

    # Create models for training
    lamba1 = 0.03
    lamba2 = 0.03
    lamba3 = 22
    encoder_train = Model([x, z_p], dis_feat_tilde, name='e')
    encoder_train.add_loss(lamba1 * kl_loss)
    encoder_train.add_loss(lamba2 * dis_nll_loss)
    encoder_train.add_loss(lamba3 * phase_loss)

    decoder_train = Model([x, z_p], [dis_x_tilde, dis_x_p], name='de')
    normalized_weight = recon_vs_gan_weight / (1. - recon_vs_gan_weight)
    decoder_train.add_loss(normalized_weight * dis_nll_loss)

    discriminator_train = Model([x, z_p], [dis_x, dis_x_tilde, dis_x_p], name='di')

    # Additional models for testing
    vae = Model(x, x_tilde, name='vae')
    vaegan = Model(x, dis_x_tilde, name='vaegan')
    vaegan_test = Model(x, z, name='vaegan_test')

    return encoder_train, decoder_train, discriminator_train, vae, vaegan, vaegan_test
