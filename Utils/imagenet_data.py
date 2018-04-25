
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
def split_classes(prm):
    #  split the labels to meta-train and meta-test
    # return the data folders paths of each split

    n_meta_train_classes = prm.n_meta_train_classes

    root_path = os.path.join(prm.data_path, 'MiniImageNet','images')
    all_label_dirs = os.listdir(root_path)

    # Take random n_meta_train_chars chars as meta-train and rest as meta-test
    random.shuffle(all_label_dirs)
    class_split = {}
    class_split['meta_train'] = all_label_dirs[:n_meta_train_classes]
    class_split['meta_test'] = all_label_dirs[n_meta_train_classes:]


    return class_split


# -------------------------------------------------------------------------------------------
#  Get task
# -------------------------------------------------------------------------------------------


def get_task(labels_in_split,  n_labels, k_train_shot, prm):
    # labels_list = labels of current split

    # Get data:
    data_dir = os.path.join(prm.data_path, 'MiniImageNet', 'images')

    # Draw random  n_labels classes from the labels in the split:
    n_tot_labels = len(labels_in_split)
    label_inds = np.random.choice(n_tot_labels, n_labels, replace=False)
    classes_names = [labels_in_split[ind] for ind in label_inds]

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
    train_dataset = image_dataset(data_dir, train_samp, train_targets)
    test_dataset = image_dataset(data_dir, test_samp, test_targets)


    return train_dataset, test_dataset


# -------------------------------------------------------------------------------------------
#  Class definition
# -------------------------------------------------------------------------------------------
class image_dataset(data.Dataset):
    def __init__(self, data_dir, samples_paths, targets):
        super(image_dataset, self).__init__()
        self.all_items = list(zip(samples_paths, targets))
        self.data_dir = data_dir


    def __getitem__(self, index):

        img = self.all_items[index][0]
        target = self.all_items[index][1]

        # Data transformations list:
        img = FilenameToPILImage(img, self.data_dir)
        img = to_tensor(img)

        # ## show  image
        # import matplotlib.pyplot as plt
        # plt.imshow(np.transpose(img.numpy(), (1, 2, 0)))
        # plt.show()

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
