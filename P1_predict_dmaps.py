import argparse
import os, glob
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from matplotlib import pyplot as plt
from PIL import Image
import models


class predict:
    def __init__(self, model_data_path):
        # Default input size
        self.height = 228
        self.width = 304
        self.channels = 3
        self.batch_size = 1

        # Create a placeholder for the input image
        self.input_node = tf.placeholder(
            tf.float32, shape=(None, self.height, self.width, self.channels)
        )

        # Construct the network
        self.net = models.ResNet50UpProj(
            {"data": self.input_node}, self.batch_size, 1, False
        )
        self.sess = tf.Session()

        # Load the converted parameters
        print("Loading the model")

        # Use to load from ckpt file
        saver = tf.train.Saver()
        saver.restore(self.sess, model_data_path)

    def __call__(self, f_img):

        # Read image
        img = Image.open(f_img)
        org_width, org_height = img.size

        img = img.resize([self.width, self.height], Image.ANTIALIAS)
        img = np.array(img).astype("float32")
        img = np.expand_dims(np.asarray(img), axis=0)

        # Evalute the network for the given image
        pred = self.sess.run(self.net.get_output(), feed_dict={self.input_node: img})
        pred = pred[0, :, :, 0]

        img = Image.fromarray(pred)
        img = img.resize([org_width, org_height], Image.BILINEAR)
        return np.asarray(img)


if __name__ == "__main__":
    # python predict.py model_files/NYU_FCRN.ckpt chair.jpg

    save_dest = "data/dmaps"
    os.system(f"mkdir -p {save_dest}")

    example_dest = "data/sample_dmap_images"
    os.system(f"mkdir -p {example_dest}")

    F_JPG = sorted(glob.glob("data/frames/*"))
    model_path = "model_files/NYU_FCRN.ckpt"

    # Predict the image
    P = None

    import pixelhouse.pixelhouse as ph

    canvas = ph.Canvas()

    for f0 in tqdm(F_JPG):
        f1 = os.path.join(save_dest, os.path.basename(f0)) + ".npy"

        if os.path.exists(f1):
            continue

        if P is None:
            P = predict(model_path)

        dmap = P(f0)
        np.save(f1, dmap)

        print(f1)
        dmap = np.clip((dmap * 50), 0, 255).astype(np.uint8)
        canvas.img = np.dstack([dmap, dmap, dmap, dmap])
        img = ph.load(f0)

        cx = ph.vstack([img, canvas])
        f_img = os.path.join(example_dest, os.path.basename(f0))
        cx.save(f_img)

        cx.resize(0.25).show(1)


        # plt.cla()
        # plt.clf()
        # ii = plt.imshow(dmap, interpolation='nearest')
        # fig.colorbar(ii)
