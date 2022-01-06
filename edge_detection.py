import argparse
import random

import PIL
import numpy as np
import scipy
from PIL import Image
from imageio import imread
from scipy import ndimage
from tensorflow.keras import layers

import data_loader
import tensorflow as tf


def create_line_image(image):
    img = ndimage.filters.convolve(image, gaussian_kernel(4))
    img, D = sobel_filters(img)
    img = non_max_suppression(img, D)
    img, weak, strong = threshold(img)
    img = hysteresis(img, weak, strong)
    return img


def gaussian_kernel(size, sigma=1):
    size = int(size) // 2
    x, y = np.mgrid[-size:size + 1, -size:size + 1]
    normal = 1 / (2.0 * np.pi * sigma ** 2)
    g = np.exp(-((x ** 2 + y ** 2) / (2.0 * sigma ** 2))) * normal
    return g


def sobel_filters(img):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    Ix = ndimage.filters.convolve(img, Kx)
    Iy = ndimage.filters.convolve(img, Ky)

    G = np.hypot(Ix, Iy)
    G = G / G.max() * 255
    theta = np.arctan2(Iy, Ix)

    return G, theta


def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M, N), dtype=np.int32)
    angle = D * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            try:
                q = 255
                r = 255

                # angle 0
                if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                    q = img[i, j + 1]
                    r = img[i, j - 1]
                # angle 45
                elif 22.5 <= angle[i, j] < 67.5:
                    q = img[i + 1, j - 1]
                    r = img[i - 1, j + 1]
                # angle 90
                elif 67.5 <= angle[i, j] < 112.5:
                    q = img[i + 1, j]
                    r = img[i - 1, j]
                # angle 135
                elif 112.5 <= angle[i, j] < 157.5:
                    q = img[i - 1, j - 1]
                    r = img[i + 1, j + 1]

                if (img[i, j] >= q) and (img[i, j] >= r):
                    Z[i, j] = img[i, j]
                else:
                    Z[i, j] = 0

            except IndexError as e:
                pass

    return Z


def threshold(img, lowThresholdRatio=0.05, highThresholdRatio=0.09):
    highThreshold = img.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio

    M, N = img.shape
    res = np.zeros((M, N), dtype=np.int32)

    weak = np.int32(25)
    strong = np.int32(255)

    strong_i, strong_j = np.where(img >= highThreshold)
    zeros_i, zeros_j = np.where(img < lowThreshold)

    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

    res[strong_i, strong_j] = strong
    res[weak_i, weak_j] = weak

    return res, weak, strong


def hysteresis(img, weak=75, strong=255):
    M, N = img.shape
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if img[i, j] == weak:
                try:
                    if ((img[i + 1, j - 1] == strong) or (img[i + 1, j] == strong) or (img[i + 1, j + 1] == strong)
                            or (img[i, j - 1] == strong) or (img[i, j + 1] == strong)
                            or (img[i - 1, j - 1] == strong) or (img[i - 1, j] == strong) or (
                                    img[i - 1, j + 1] == strong)):
                        img[i, j] = strong
                    else:
                        img[i, j] = 0
                except IndexError as e:
                    pass
    return img


def define_parser():
    p = argparse.ArgumentParser()
    p.add_argument('--i', type=str, required=True, help='Image to edge detect')
    p.add_argument('--o', type=str, default=28, help='Folder in which to store output')
    p.add_argument('--a', type=int, default=4, help='Number of augmentations for each image')
    p.add_argument('--s', type=int, default=4, help='Shape of image')
    return p


def add_augmentation(img):
    augment = tf.keras.Sequential([
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.1, fill_mode="constant", fill_value=0),
        layers.experimental.preprocessing.RandomZoom((0, 0.5), fill_mode="constant", fill_value=0)
    ])
    augmented_img = augment(img).numpy()
    return augmented_img


def format_image_for_output(o_i, n_i, shape):
    o_i = Image.fromarray(o_i).convert('RGB').resize(shape)
    n_i = Image.fromarray(n_i).convert('RGB').resize(shape)
    merged_im = Image.new('RGB', (shape[0] * 2, shape[1]))
    merged_im.paste(o_i, (0, 0))
    merged_im.paste(n_i, (shape[0], 0))
    return merged_im


if __name__ == '__main__':
    parser = define_parser()
    args = parser.parse_args()
    img_path = args.i

    for counter in range(args.a):
        orig_img = add_augmentation(np.array(Image.open(img_path).convert('RGB')))
        # img = orig_img.convert("L")
        img = np.dot(orig_img[..., :3], [0.2989, 0.5870, 0.1140])
        new_img = create_line_image(img)

        format_image_for_output(orig_img, new_img, (args.s, args.s)).save(args.o + "_" + str(counter) + ".png")
