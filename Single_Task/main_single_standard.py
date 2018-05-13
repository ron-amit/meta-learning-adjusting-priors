
from __future__ import absolute_import, division, print_function

import argparse
import os
import torch
import torch.optim as optim
from Utils import data_gen
from Utils.common import set_random_seed, save_model_state, create_result_dir, save_run_data
from Single_Task import learn_single_standard
from Data_Path import get_data_path

torch.backends.cudnn.benchmark = True  # For speed improvement with models with fixed-length inputs
# -------------------------------------------------------------------------------------------
#  Set Parameters
# -------------------------------------------------------------------------------------------

parser = argparse.ArgumentParser()

# ----- Run Parameters ---------------------------------------------#

parser.add_argument('--run-name', type=str, help='Name of dir to save results in (if empty, name by time)',
                    default='')

parser.add_argument('--seed', type=int,  help='random seed',
                    default=1)

parser.add_argument('--test-batch-size',type=int,  help='input batch size for testing (reduce if memory is limited)',
                    default=128)

# ----- Task Parameters ---------------------------------------------#

parser.add_argument('--data-source', type=str, help="Data: 'MNIST' / 'CIFAR10' / Omniglot / SmallImageNet",
                    default='MNIST')

parser.add_argument('--data-transform', type=str, help="Data transformation:  'None' / 'Permute_Pixels' / 'Permute_Labels'/ Shuffled_Pixels ",
                    default='None')

parser.add_argument('--limit_train_samples', type=int,
                    help='Upper limit for the number of training samples (0 = unlimited)',
                    default=0)

# N-Way K-Shot Parameters:
parser.add_argument('--N_Way', type=int, help='Number of classes in a task (for Omniglot)',
                    default=5)

parser.add_argument('--K_Shot_MetaTrain', type=int,
                    help='Number of training sample per class in meta-training in N-Way K-Shot data sets',
                    default=100)  # Note:  test samples are the rest of the data

parser.add_argument('--K_Shot_MetaTest', type=int,
                    help='Number of training sample per class in meta-testing in N-Way K-Shot data sets',
                    default=100)  # Note:  test samples are the rest of the data

# SmallImageNet Parameters:
parser.add_argument('--n_meta_train_classes', type=int,
                    help='For SmallImageNet: how many catagories to use for meta-training',
                    default=200)

# Omniglot Parameters:
parser.add_argument('--chars_split_type', type=str,
                    help='how to split the Omniglot characters  - "random" / "predefined_split"',
                    default='random')

parser.add_argument('--n_meta_train_chars', type=int,
                    help='For Omniglot: how many characters to use for meta-training, if split type is random',
                    default=1200)

# ----- Algorithm Parameters ---------------------------------------------#

parser.add_argument('--loss-type', type=str, help="Data: 'CrossEntropy' / 'L2_SVM'",
                    default='CrossEntropy')

parser.add_argument('--model-name', type=str, help="Define model type (hypothesis class)'",
                    default='OmConvNet')  # ConvNet3 / 'FcNet3' / 'OmConvNet'

parser.add_argument('--batch-size', type=int, help='input batch size for training',
                    default=128)

parser.add_argument('--num-epochs', type=int, help='number of epochs to train',
                    default=50)

parser.add_argument('--lr', type=float, help='initial learning rate',
                    default=1e-3)
# -------------------------------------------------------------------------------------------

prm = parser.parse_args()
prm.data_path = get_data_path()
set_random_seed(prm.seed)
create_result_dir(prm)

#  Define optimizer:
prm.optim_func, prm.optim_args = optim.Adam,  {'lr': prm.lr} #  'weight_decay':5e-4
# prm.optim_func, prm.optim_args = optim.SGD, {'lr': prm.lr, 'momentum': 0.9, 'weight_decay':5e-4}

# Learning rate decay schedule:
# prm.lr_schedule = {'decay_factor': 0.1, 'decay_epochs': [150, 225]}
prm.lr_schedule = {}  # No decay

# Generate task data set:
task_generator = data_gen.Task_Generator(prm)
data_loader = task_generator.get_data_loader(prm, limit_train_samples=prm.limit_train_samples)

# -------------------------------------------------------------------------------------------
#  Run learning
# -------------------------------------------------------------------------------------------

test_err, model = learn_single_standard.run_learning(data_loader, prm)

# save final learned weights
f_name = 'final_weights.pt'
f_path = os.path.join(prm.result_dir, f_name)
f_path = save_model_state(model, f_path)
print('Trained model saved in ' + f_path)

save_run_data(prm, {'test_err': test_err})