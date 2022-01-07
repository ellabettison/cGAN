import argparse
import datetime
import os

import skimage
import tensorflow.python.ops.summary_ops_v2
from PIL import Image, ImageCms
from tensorflow.keras.initializers import RandomNormal
import tensorflow.keras as tf

from data_loader import DataLoader
import numpy as np
from tensorflow.keras.layers import Conv2D, LeakyReLU, BatchNormalization, UpSampling2D, Dropout, Concatenate, \
    Conv2DTranspose
from tensorflow.keras import Input, Model, backend
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())
log_device_placement = True


def conv2d_layer(layer_inp, filters, batch_norm=True, strides=2):
    c = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same', use_bias=False)(layer_inp)
    if batch_norm:
        c = BatchNormalization(momentum=0.8)(c)
    c = LeakyReLU(alpha=0.2)(c)
    return c


def deconv2d_layer(layer_input, skip_input, filters, dropout_rate=0.0, upsample_size=2, strides=1):
    d = Conv2DTranspose(filters, kernel_size=kernel_size, strides=2, padding='same', activation='relu')(layer_input)
    if dropout_rate:
        d = Dropout(dropout_rate)(d)
    d = BatchNormalization(momentum=0.8)(d)
    d = Concatenate()([d, skip_input])
    return d


def define_discriminator():
    init = RandomNormal(stddev=0.02)

    img_A = Input(shape=img_shape)
    img_B = Input(shape=img_shape)

    combined_imgs = Concatenate(axis=-1)([img_A, img_B])

    d1 = conv2d_layer(combined_imgs, dis_filters, batch_norm=False)
    d2 = conv2d_layer(d1, dis_filters * 2)
    d3 = conv2d_layer(d2, dis_filters * 4)
    d4 = conv2d_layer(d3, dis_filters * 8)

    validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

    model = Model([img_A, img_B], validity)
    opt = Adam(args.lr, beta_1=0.5)

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


def define_generator():
    init = RandomNormal(stddev=0.02)
    inp = Input(shape=img_shape)

    # downsampling layers
    d1 = conv2d_layer(inp, gen_filters, batch_norm=False)
    d2 = conv2d_layer(d1, gen_filters * 2)
    d3 = conv2d_layer(d2, gen_filters * 4)
    d4 = conv2d_layer(d3, gen_filters * 8)
    d5 = conv2d_layer(d4, gen_filters * 8)
    d6 = conv2d_layer(d5, gen_filters * 8)
    d7 = conv2d_layer(d6, gen_filters * 8)

    # upsampling layers
    u1 = deconv2d_layer(d7, d6, gen_filters * 8, dropout_rate=0.5)
    u2 = deconv2d_layer(u1, d5, gen_filters * 8, dropout_rate=0.5)
    u3 = deconv2d_layer(u2, d4, gen_filters * 8)
    u4 = deconv2d_layer(u3, d3, gen_filters * 4)
    u5 = deconv2d_layer(u4, d2, gen_filters * 2)
    u6 = deconv2d_layer(u5, d1, gen_filters)

    output_img = Conv2DTranspose(channels, kernel_size=kernel_size, strides=(2, 2), padding='same', kernel_initializer=init)(u6)

    return Model(inp, output_img)


def define_gan(g_model, d_model, opt):
    img_A = Input(shape=img_shape)
    img_B = Input(shape=img_shape)

    fake_A = g_model(img_B)

    d_model.trainable = False

    valid = d_model([fake_A, img_B])

    # should create valid output and recreate A?
    gan = Model(inputs=[img_A, img_B], outputs=[valid, fake_A])
    # mae = mean absolute error
    # loss_weights - weight
    # mae
    gan.compile(loss=[tf.losses.BinaryCrossentropy(from_logits=True), bw_l1_loss], loss_weights=[1, l1_balance], optimizer=opt)
    return gan


def bw_l1_loss(y_true, y_pred):
    y_true_bw = tensorflow.image.rgb_to_grayscale(y_true) #np.dot(y_true.numpy()[..., :3], [0.2989, 0.5870, 0.1140])
    y_pred_bw = tensorflow.image.rgb_to_grayscale(y_pred) #np.dot(y_pred[..., :3], [0.2989, 0.5870, 0.1140])
    return tf.losses.mae(y_true_bw, y_pred_bw)


def train(g_model, d_model, gan_model):
    start_time = datetime.datetime.now()

    valid_outputs = np.ones((batch_size,) + disc_patch)
    fake_ouputs = np.zeros((batch_size,) + disc_patch)

    for epoch in range(epochs):
        for batch_i, (imgs_A, imgs_B) in enumerate(data_loader.load_batch(batch_size)):
            # train discriminator
            fake_A = g_model.predict(imgs_B)

            d_loss_real = d_model.train_on_batch([imgs_A, imgs_B], valid_outputs)
            d_loss_fake = d_model.train_on_batch([fake_A, imgs_B], fake_ouputs)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # train generator
            g_loss = gan_model.train_on_batch([imgs_A, imgs_B], [valid_outputs, imgs_A])

            elapsed_time = datetime.datetime.now() - start_time
            # Plot the progress
            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs,
                                                                                                  batch_i,
                                                                                                  data_loader.n_batches,
                                                                                                  d_loss[0],
                                                                                                  100 * d_loss[1],
                                                                                                  g_loss[0],
                                                                                                  elapsed_time))

            if batch_i % sample_interval == 0:
                sample_images(epoch, batch_i, g_model)
            if epoch % model_save_interval == 0 and batch_i == 0:
                g_model.save("models/model%d_%d" % (epoch, batch_i))


def sample_images(epoch, batch_i, g_model):
    print("saving figs")
    os.makedirs('%s/%s' % (output_loc, dataset_name), exist_ok=True)
    r, c = 3, 3

    imgs_A, imgs_B = data_loader.load_data(batch_size=3, is_testing=True)
    fake_A = g_model.predict(imgs_B)

    gen_imgs = np.concatenate([imgs_B, fake_A, imgs_A])

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    titles = ['Condition', 'Generated', 'Original']
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt])
            axs[i, j].set_title(titles[i])
            axs[i, j].axis('off')
            cnt += 1
    # fig.savefig("images/%s/%d_%d.png" % (dataset_name, epoch, batch_i))
    plt.show()
    plt.close()


def define_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--n_classes', type=int, default=10, help='Number of classes to generate')
    p.add_argument('--image_size', type=int, default=128, help='Image width and height in pixels')
    p.add_argument('--lr', type=float, default=0.0002, help='Learning rate')
    p.add_argument('--batch_size', type=int, default=4, help='Batch size')
    p.add_argument('--l1_balance', type=int, default=100, help='L1 balance')
    p.add_argument('--o', type=str, default="images", help='L1 balance')
    return p


if __name__ == '__main__':
    parser = define_parser()
    args = parser.parse_args()
    n_classes = args.n_classes

    gen_filters = 64
    dis_filters = 64
    channels = 3
    img_size = 128
    kernel_size = 4
    img_shape = (img_size, img_size, channels)
    epochs = 20
    batch_size = args.batch_size #1
    sample_interval = 10
    model_save_interval = 1
    l1_balance = args.l1_balance
    output_loc = args.o

    patch = int(img_size / 2 ** 4)
    disc_patch = (patch, patch, 1)

    dataset_name = 'pokemon'
    data_loader = DataLoader(dataset_name=dataset_name,
                             img_res=(img_size, img_size))

    opt = Adam(args.lr, 0.5)

    g_model = define_generator()
    d_model = define_discriminator()
    gan = define_gan(g_model, d_model, opt)
    train(g_model, d_model, gan)
