import random
from glob import glob
import numpy as np
import scipy
import skimage
from PIL import Image, ImageCms
from skimage import color, io


class DataLoader:
    def __init__(self, dataset_name, img_res=(128, 128)):
        self.dataset_name = dataset_name
        self.img_res = img_res

        # Convert to Lab colourspace
        srgb_p = ImageCms.createProfile("sRGB")
        lab_p = ImageCms.createProfile("LAB")

        self.rgb2lab = ImageCms.buildTransformFromOpenProfiles(srgb_p, lab_p, "RGB", "LAB")
        self.lab2rgb = ImageCms.buildTransformFromOpenProfiles(lab_p, srgb_p, "LAB", "RGB")

    def load_data(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "test"
        path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))

        batch_images = np.random.choice(path, size=batch_size)

        imgs_A = []
        imgs_B = []
        for img_path in batch_images:
            img = self.imread(img_path)

            h, w, _ = img.shape
            _w = int(w / 2)
            img_A, img_B = img[:, :_w, :], img[:, _w:, :]

            img_A = np.array(Image.fromarray(img_A).resize(self.img_res))
            # img_A = scipy.misc.imresize(img_A, self.img_res)
            # img_B = scipy.misc.imresize(img_B, self.img_res)
            img_B = np.array(Image.fromarray(img_B).resize(self.img_res))

            # If training => do random flip
            if not is_testing and np.random.random() < 0.5:
                img_A = np.fliplr(img_A)
                img_B = np.fliplr(img_B)

            imgs_A.append(img_A)
            imgs_B.append(img_B)

        imgs_A = np.array(imgs_A) / 127.5 - 1.
        imgs_B = np.array(imgs_B) / 127.5 - 1.

        return imgs_A, imgs_B

    def load_batch(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "val"
        path = glob('./datasets/%s/%s/*' % (self.dataset_name, data_type))
        random.shuffle(path)

        self.n_batches = int(len(path) / batch_size)

        for i in range(self.n_batches - 1):
            batch = path[i * batch_size:(i + 1) * batch_size]
            imgs_A, imgs_B = [], []
            for img in batch:
                img = self.imread(img)
                h, w, _ = img.shape

                half_w = int(w / 2)
                img_A = img[:, :half_w, :]
                img_B = img[:, half_w:, :]

                img_A = np.array(Image.fromarray(img_A).resize(self.img_res))
                # img_A = scipy.misc.imresize(img_A, self.img_res)
                # img_B = scipy.misc.imresize(img_B, self.img_res)
                img_B = np.array(Image.fromarray(img_B).resize(self.img_res))

                if not is_testing and np.random.random() > 0.5:
                    img_A = np.fliplr(img_A)
                    img_B = np.fliplr(img_B)

                imgs_A.append(img_A)
                imgs_B.append(img_B)

            imgs_A = np.array(imgs_A) / 127.5 - 1.
            imgs_B = np.array(imgs_B) / 127.5 - 1.

            yield imgs_A, imgs_B

    def imread(self, path):
        return np.array(Image.open(path).convert('RGB'))
        # return np.array(color.rgb2lab(Image.open(path).convert('RGB')))
        # return imageio.imread(path, as_gray=False, pilmode="RGB").astype(np.float)
        # return color.rgb2lab(io.imread(path, pilmode='RGB'))
