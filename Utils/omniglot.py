
from __future__ import absolute_import, division, print_function

import os
import os.path
import errno
import random
from PIL import Image
import torch.utils.data as data
from torchvision.transforms.functional import to_tensor
import numpy as np
import torch


# Based on code from:
# https://github.com/katerakelly/pytorch-maml/blob/master/src/task.py
# https://github.com/pytorch/vision/pull/46

# data set: https://github.com/brendenlake/omniglot


# -------------------------------------------------------------------------------------------
# Auxiliary function
# -------------------------------------------------------------------------------------------
def get_all_char_paths(data_dir):
    languages = os.listdir(data_dir)
    predefined_split_dir = os.path.split(data_dir)[-1]
    chars = []
    for lang in languages:
        chars += [os.path.join(predefined_split_dir, lang, x)
                  for x in os.listdir(os.path.join(data_dir, lang))]
    return chars

# -------------------------------------------------------------------------------------------
#  Create meta-split of characters
# -------------------------------------------------------------------------------------------
def split_chars(data_path, chars_split_type, n_meta_train_chars):
    #  split the characters to meta-train and meta-test
    # return the data folders paths of each split

    split_names = ['meta_train', 'meta_test']
    chars_splits = {}

    # Get data:
    root_path = os.path.join(data_path, 'Omniglot')
    predefined_splits_dirs = maybe_download(root_path)

    if chars_split_type == 'predefined_split':

        for split_name in split_names:
            data_dir = predefined_splits_dirs[split_name]
            chars_splits[split_name] = get_all_char_paths(data_dir)

    elif chars_split_type == 'random':
        # Get all chars dirs (don't care about pre-defined splits):
        chars = sum([get_all_char_paths(predefined_splits_dirs[split_name])
                 for split_name in split_names], [])

        # Take random n_meta_train_chars chars as meta-train and rest as meta-test
        random.shuffle(chars)
        chars_splits['meta_train'] = chars[:n_meta_train_chars]
        chars_splits['meta_test'] = chars[n_meta_train_chars:]

    else:
        raise ValueError('Unrecognized split_type')

    return chars_splits

# -------------------------------------------------------------------------------------------
#  Get task
# -------------------------------------------------------------------------------------------

def get_task(chars, root_path, n_labels, k_train_shot, final_input_trans=None, target_transform=None):

    '''
    Samples a N-way k-shot learning task (classification to N classes,
     k training samples per class) from the Omniglot dataset.

     -  n_labels = number of labels (chars) in the task.
     - chars =   list of chars dirs  for current meta-split
     - k_train_shot - sample this many training examples from each char class,
                      rest of the char examples will be in the test set.

      e.g:
    data_loader = get_omniglot_task(prm, meta_split='meta_train', n_labels=5, k_train_shot=10)
    '''

    # Get data:
    data_dir = os.path.join(root_path, 'Omniglot', 'processed')

    # Sample n_labels classes:
    n_tot_chars = len(chars)
    char_inds = np.random.choice(n_tot_chars, n_labels, replace=False)
    classes_names = [chars[ind] for ind in char_inds]

    train_samp = []
    test_samp = []
    train_targets = []
    test_targets = []

    for i_label in range(n_labels):

        class_dir = classes_names[i_label]
        # First get all instances of that class
        all_class_samples = [os.path.join(class_dir, x) for x in os.listdir(os.path.join(data_dir, class_dir))]
        if not k_train_shot:
            k_train_shot = len(all_class_samples)
        # Sample k_train_shot instances randomly each for train
        random.shuffle(all_class_samples)
        cls_train_samp = all_class_samples[:k_train_shot]
        train_samp += cls_train_samp
        # Rest go to test set:
        cls_test_samp = all_class_samples[k_train_shot+1:]
        test_samp += cls_test_samp

        # Targets \ labels:
        train_targets += [i_label] * len(cls_train_samp)
        test_targets += [i_label] * len(cls_test_samp)

    # Create the dataset object:
    train_dataset = omniglot_dataset(data_dir, train_samp, train_targets, final_input_trans, target_transform)
    test_dataset = omniglot_dataset(data_dir, test_samp, test_targets, final_input_trans, target_transform)


    return train_dataset, test_dataset

# -------------------------------------------------------------------------------------------
#  Class definition
# -------------------------------------------------------------------------------------------
class omniglot_dataset(data.Dataset):
    def __init__(self, data_dir, samples_paths, targets, final_input_trans=None, target_transform=None):
        super(omniglot_dataset, self).__init__()
        self.all_items = list(zip(samples_paths, targets))
        self.final_input_trans = final_input_trans
        self.target_transform = target_transform
        self.data_dir = data_dir


    def __getitem__(self, index):

        img = self.all_items[index][0]
        target = self.all_items[index][1]

        # Data transformations list:
        img = FilenameToPILImage(img, self.data_dir)
        # Re-size to 28x28  (to compare to prior papers)
        img = img.resize((28, 28), resample=Image.LANCZOS)

        img = to_tensor(img)
        img = img.mean(dim=0).unsqueeze_(0)  # RGB -> gray scale
        img = 1.0 - img  # Switch background to 0 and letter to 1

        # show  image
        # import matplotlib.pyplot as plt
        # plt.imshow(img.numpy()[0])
        # plt.show()

        final_input_trans = self.final_input_trans
        if final_input_trans:
            if isinstance(final_input_trans, list):
                final_input_trans = final_input_trans[0]
            img = final_input_trans(img)

        if self.target_transform:
            for trasns in self.target_transform:
                target = trasns(target)

        return img, target

    def __len__(self):
        return len(self.all_items)


# -------------------------------------------------------------------------------------------
#  Auxiliary functions
# -------------------------------------------------------------------------------------------

def FilenameToPILImage(filename, data_dir):
    """
    Load a PIL RGB Image from a filename.
    """
    file_path = os.path.join(data_dir, filename)
    img=Image.open(file_path).convert('RGB')
    # img.save("tmp.png") # debug
    return img

def check_exists(splits_dirs):
    paths = list(splits_dirs.values())
    return all([os.path.exists(path) for path in paths])

def maybe_download(root):
    from six.moves import urllib
    import zipfile

    processed_path = os.path.join(root, 'processed')
    splits_dirs = {'meta_train':  os.path.join(processed_path, 'images_background'),
                 'meta_test': os.path.join(processed_path, 'images_evaluation')}
    if check_exists(splits_dirs):
        return splits_dirs

    # download files
    data_urls = [
        'https://github.com/brendenlake/omniglot/raw/master/python/images_background.zip',
        'https://github.com/brendenlake/omniglot/raw/master/python/images_evaluation.zip']

    raw_folder = 'raw'
    processed_folder = 'processed'
    try:
        os.makedirs(os.path.join(root, raw_folder))
        os.makedirs(os.path.join(root, processed_folder))
    except OSError as e:
        if e.errno == errno.EEXIST:
            pass
        else:
            raise

    for url in data_urls:
        print('== Downloading ' + url)
        data = urllib.request.urlopen(url)
        filename = url.rpartition('/')[2]
        file_path = os.path.join(root, raw_folder, filename)
        with open(file_path, 'wb') as f:
            f.write(data.read())
        file_processed = os.path.join(root, processed_folder)
        print("== Unzip from "+file_path+" to "+file_processed)
        zip_ref = zipfile.ZipFile(file_path, 'r')
        zip_ref.extractall(file_processed)
        zip_ref.close()
    print("Download finished.")
    return splits_dirs

# ----------------   Resize images to 28x28
# """
# Usage instructions:
#     First download the omniglot dataset
#     and put the contents of both images_background and images_evaluation in data/omniglot/ (without the root folder)
#     Then, run the following:
#     cd data/
#     cp -r omniglot/* omniglot_resized/
#     cd omniglot_resized/
#     python resize_images.py
# """
# from PIL import Image
# import glob
#
# image_path = '*/*/'
#
# all_images = glob.glob(image_path + '*')
#
# i = 0
#
# for image_file in all_images:
#     im = Image.open(image_file)
#     im = im.resize((28,28), resample=Image.LANCZOS)
#     im.save(image_file)
#     i += 1
#
#     if i % 200 == 0:
#         print(i)
