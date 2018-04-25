"""
based on:
https://github.com/cbfinn/maml/blob/master/data/miniImagenet/proc_images.py

Step 1:
    Download ilsvrc2012_img_train.tar from
     http://www.image-net.org/challenges/LSVRC/2012/nonpub-downloads
     and place in <data_path>/miniImagenet/images

Step 2:  Run the following commands that extract and DELETES THE ORIGINAL tar file:
    go to images dir
    $ tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
    $ find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
    // Make sure to check the completeness of the decompression, you should have 1,281,167 images in train folder


Step 3:
run this script which:
1. resizes the images to 84x84


"""
from __future__ import absolute_import, division, print_function
import csv
import glob
import os

from PIL import Image

from Data_Path import get_data_path

input_dir = os.path.join(get_data_path(), 'MiniImageNet')

path_to_images = os.path.join(input_dir, 'images')

all_images = glob.glob(path_to_images + '/*/*')

n_images = len(all_images)

# Resize images
for i, image_file in enumerate(all_images):
    try:
        im = Image.open(image_file)
        if not (im.height == 84 and im.width == 84):
            im = im.resize((84, 84), resample=Image.LANCZOS)
            im.save(image_file)
    except OSError:
        print('Failed on ' + image_file)

    if i % 1000 == 0:
        print('{:.3}%'.format(100 * i / n_images))

