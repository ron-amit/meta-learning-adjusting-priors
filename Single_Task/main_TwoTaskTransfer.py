from __future__ import absolute_import, division, print_function

import argparse
from copy import deepcopy
import numpy as np
import torch
import torch.optim as optim

from Single_Task import learn_single_Bayes, learn_single_standard
from Utils.data_gen import Task_Generator
from Utils.common import write_to_log, set_random_seed, create_result_dir, save_run_data
from Data_Path import get_data_path

torch.backends.cudnn.benchmark = True # For speed improvement with convnets with fixed-length inputs - https://discuss.pytorch.org/t/pytorch-performance/3079/7


# -------------------------------------------------------------------------------------------
#  Set Parameters
# -------------------------------------------------------------------------------------------

# Training settings
parser = argparse.ArgumentParser()

parser.add_argument('--Experiment_Name', type=str, help='Permute_Labels / Shuffled_Pixels',
                    default='Permute_Labels')

parser.add_argument('--seed', type=int,  help='random seed',
                    default=1)

parser.add_argument('--data-source', type=str, help="Data: 'MNIST' / 'Sinusoid' ",
                    default='MNIST')

parser.add_argument('--loss-type', type=str, help="Data: 'CrossEntropy' / 'L2_SVM'",
                    default='CrossEntropy')

parser.add_argument('--batch-size', type=int, help='input batch size for training',
                    default=128)

parser.add_argument('--num-epochs', type=int, help='number of epochs to train',
                    default=200)  # 200

parser.add_argument('--lr', type=float, help='initial learning rate',
                    default=1e-3)

parser.add_argument('--test-batch-size',type=int,  help='input batch size for testing',
                    default=1000)

prm = parser.parse_args()
prm.data_path = get_data_path()
set_random_seed(prm.seed)


if prm.Experiment_Name == 'Permute_Labels':
    prm.run_name = 'TwoTaskTransfer_permuted_labels'
    prm.data_transform = 'Permute_Labels'
    prm.model_name = 'ConvNet3'
    freeze_description = 'freeze lower layers'
    not_freeze_list = ['fc_out']
    freeze_list = None

elif prm.Experiment_Name == 'Shuffled_Pixels':
    n_pixels_shuffles = 200
    prm.run_name = 'TwoTaskTransfer_shuffled_pixels' + str(n_pixels_shuffles) + '_v2'
    prm.data_transform = 'Shuffled_Pixels'
    prm.n_pixels_shuffles = n_pixels_shuffles
    prm.model_name = 'FcNet3'
    # freeze_description = 'freeze output layer'
    # freeze_list = ['fc_out']
    # not_freeze_list = None
    freeze_description = 'freeze all layers except first'
    not_freeze_list = ['fc1']
    freeze_list = None

else:
    raise ValueError('Unrecognized Experiment_Name')

create_result_dir(prm)

n_experiments = 20

limit_train_samples = 2000


#  Define optimizer:
prm.optim_func, prm.optim_args = optim.Adam,  {'lr': prm.lr}
# optim_func, optim_args = optim.SGD, {'lr': prm.lr, 'momentum': 0.9}

# Learning rate decay schedule:
# lr_schedule = {'decay_factor': 0.1, 'decay_epochs': [10]}
prm.lr_schedule = {} # No decay


# # For L2 regularization experiment:
prm_reg = deepcopy(prm)
prm_reg.optim_args['weight_decay'] = 1e-3

# For freeze lower layers experiment:
prm_freeze = deepcopy(prm)
if not_freeze_list:
    prm_freeze.not_freeze_list = not_freeze_list
if freeze_list:
    prm_freeze.freeze_list = freeze_list

# For bayes experiment -
prm.log_var_init = {'mean': -10, 'std': 0.1} # The initial value for the log-var parameter (rho) of each weight
prm.n_MC = 1 # Number of Monte-Carlo iterations
prm.test_type = 'MaxPosterior' # 'MaxPosterior' / 'MajorityVote'


# -------------------------------------------------------------------------------------------
#  Run experiments
# -------------------------------------------------------------------------------------------

task_generator = Task_Generator(prm)

test_err_orig = np.zeros(n_experiments)
test_err_scratch = np.zeros(n_experiments)
test_err_scratch_bayes = np.zeros(n_experiments)
test_err_transfer = np.zeros(n_experiments)
test_err_scratch_reg = np.zeros(n_experiments)
test_err_freeze = np.zeros(n_experiments)

for i in range(n_experiments):
    write_to_log('--- Experiment #{} out of {}'.format(i+1, n_experiments), prm)

    # Generate the task #1 data set:
    task1_data = task_generator.get_data_loader(prm)
    n_samples_orig = task1_data['n_train_samples']

    #  Run learning of task 1
    write_to_log('--- Standard learning of task #1', prm)
    test_err_orig[i], transferred_model = learn_single_standard.run_learning(task1_data, prm)

    # Generate the task 2 data set:

    write_to_log('--- Generating task #2 with at most {} samples'.format(limit_train_samples), prm)
    task2_data = task_generator.get_data_loader(prm, limit_train_samples = limit_train_samples)

    # #  Run learning of task 2 from scratch:
    # write_to_log('--- Standard learning of task #2 from scratch', prm)
    # test_err_scratch[i], _ = learn_single_standard.run_learning(task2_data, prm, verbose=0)
    #
    # #  Run Bayesian-learning of task 2 from scratch:
    # write_to_log('---- Bayesian learning of task #2 from scratch', prm)
    # test_err_scratch_bayes[i], _ = learn_single_Bayes.run_learning(task2_data, prm, verbose=0)
    #
    # #  Run learning of task 2 using transferred initial point:
    # write_to_log('--- Standard learning of task #2 using transferred weights as initial point', prm)
    # test_err_transfer[i], _ = learn_single_standard.run_learning(task2_data, prm, initial_model=transferred_model, verbose=0)

    #  Run learning of task 2 using transferred initial point + freeze some layers:
    write_to_log('--- Standard learning of task #2 using transferred weights as initial point + ' + freeze_description, prm_freeze)
    test_err_freeze[i], _ = learn_single_standard.run_learning(task2_data, prm_freeze, initial_model=transferred_model, verbose=0)
    #
    # #  Run learning of task 2 from scratch + weight regularization:
    # write_to_log('--- Standard learning of task #2 from scratch', prm_reg)
    # test_err_scratch_reg[i], _ = learn_single_standard.run_learning(task2_data, prm_reg, verbose=0)


# -------------------------------------------------------------------------------------------
#  Print Results
# -------------------------------------------------------------------------------------------

write_to_log('--- Final Results: ', prm)
write_to_log('Averaging of {} experiments...'.format(n_experiments), prm)


write_to_log('Standard learning of task #1 ({} samples), average test error: {:.3}%, STD: {:.3}%'.
             format(n_samples_orig, 100*test_err_orig.mean(), 100*test_err_orig.std()), prm)

write_to_log('Standard learning of task #2  (at most {} samples)'
             ' from scratch, average test error: {:.3}%, STD: {:.3}%'.
             format(limit_train_samples, 100*test_err_scratch.mean(), 100*test_err_scratch.std()), prm)

write_to_log('Bayesian learning of task #2  (at most {} samples)'
             ' from scratch, average test error: {:.3}%, STD: {:.3}%'.
             format(limit_train_samples, 100*test_err_scratch_bayes.mean(), 100*test_err_scratch.std()), prm)


write_to_log('Standard learning of task #2  (at most {} samples) '
             'from scratch with L2 regularizer, average test error: {:.3}%, STD: {:.3}%'.
             format(limit_train_samples, 100*test_err_scratch_reg.mean(), 100*test_err_scratch_reg.std()), prm_reg)

write_to_log('Standard learning of task #2  (at most {} samples)'
             ' using transferred weights as initial point, average test error: {:.3}%, STD: {:.3}%'.
             format(limit_train_samples, 100*test_err_transfer.mean(), 100*test_err_transfer.std()), prm)

write_to_log('Standard learning of task #2  (at most {} samples) using transferred weights as initial point '
             ' + {}, average test error: {:.3}%, STD: {:.3}%'.
             format(limit_train_samples, freeze_description, 100*test_err_freeze.mean(), 100*test_err_freeze.std()), prm_freeze)


# -------------------------------------------------------------------------------------------
#  Save Results
# -------------------------------------------------------------------------------------------

save_run_data(prm, {'test_err_orig': test_err_orig, 'test_err_scratch': test_err_scratch, 'test_err_scratch_bayes': test_err_scratch_bayes,
                    'test_err_transfer': test_err_transfer, 'test_err_freeze':test_err_freeze, 'test_err_scratch_reg': test_err_scratch_reg})
