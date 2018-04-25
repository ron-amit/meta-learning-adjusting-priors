
from __future__ import absolute_import, division, print_function

import argparse
import timeit, time, os
import numpy as np
import torch
import torch.optim as optim

from PriorMetaLearning import meta_test_Bayes, meta_train_Bayes_finite_tasks, meta_train_Bayes_infinite_tasks
from Data_Path import get_data_path
from Models.stochastic_models import get_model
from Utils.data_gen import Task_Generator
from Utils.common import save_model_state, load_model_state, create_result_dir, set_random_seed, write_to_log, save_run_data
from PriorMetaLearning.Analyze_Prior import run_prior_analysis

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

parser.add_argument('--mode', type=str, help='MetaTrain or LoadMetaModel',
                    default='MetaTrain')   # 'MetaTrain'  \ 'LoadMetaModel'

parser.add_argument('--load_model_path', type=str, help='set the path to pre-trained model, in case it is loaded (if empty - set according to run_name)',
                    default='')

parser.add_argument('--test-batch-size',type=int,  help='input batch size for testing (reduce if memory is limited)',
                    default=128)

parser.add_argument('--n_test_tasks', type=int,
                    help='Number of meta-test tasks for meta-evaluation (how many tasks to average in final result)',
                    default=10)

# ----- Task Parameters ---------------------------------------------#

parser.add_argument('--data-source', type=str, help="Data: 'MNIST' / 'CIFAR10' / Omniglot / SmallImageNet",
                    default='MNIST')

parser.add_argument('--n_train_tasks', type=int, help='Number of meta-training tasks (0 = infinite)',
                    default=5)

parser.add_argument('--data-transform', type=str, help="Data transformation:  'None' / 'Permute_Pixels' / 'Permute_Labels'/ Shuffled_Pixels ",
                    default='Permute_Labels')

parser.add_argument('--n_pixels_shuffles', type=int, help='In case of "Shuffled_Pixels": how many pixels swaps',
                    default=200)

parser.add_argument('--limit_train_samples_in_test_tasks', type=int,
                    help='Upper limit for the number of training samples in the meta-test tasks (0 = unlimited)',
                    default=2000)

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
                    help='For SmallImageNet: how many categories are available for meta-training',
                    default=500)

# Omniglot Parameters:
parser.add_argument('--chars_split_type', type=str,
                    help='how to split the Omniglot characters  - "random" / "predefined_split"',
                    default='random')

parser.add_argument('--n_meta_train_chars'
                    , type=int, help='For Omniglot: how many characters to use for meta-training, if split type is random',
                    default=1200)

# ----- Algorithm Parameters ---------------------------------------------#

parser.add_argument('--complexity_type', type=str,
                    help=" The learning objective complexity type",
                    default='NewBoundSeeger')  #  'NoComplexity' /  'Variational_Bayes' / 'PAC_Bayes_Pentina'   NewBoundMcAllaster / NewBoundSeeger'"

# parser.add_argument('--override_eps_std', type=float,
#                     help='For debug: set the STD of epsilon variable for re-parametrization trick (default=1.0)',
#                     default=1.0)

parser.add_argument('--loss-type', type=str, help="Loss function",
                    default='CrossEntropy') #  'CrossEntropy' / 'L2_SVM'

parser.add_argument('--model-name', type=str, help="Define model type (hypothesis class)'",
                    default='ConvNet3')  # OmConvNet / 'FcNet3' / 'ConvNet3'

parser.add_argument('--batch-size', type=int, help='input batch size for training',
                    default=128)

parser.add_argument('--n_meta_train_epochs', type=int, help='number of epochs to train',
                    default=150)

parser.add_argument('--n_inner_steps', type=int,
                    help='For infinite tasks case, number of steps for training per meta-batch of tasks',
                    default=50)  #

parser.add_argument('--n_meta_test_epochs', type=int, help='number of epochs to train',
                    default=200)  #

parser.add_argument('--lr', type=float, help='initial learning rate',
                    default=1e-3)

parser.add_argument('--meta_batch_size', type=int, help='Maximal number of tasks in each meta-batch',
                    default=5)
# -------------------------------------------------------------------------------------------

prm = parser.parse_args()
prm.data_path = get_data_path()
set_random_seed(prm.seed)
create_result_dir(prm)


# Weights initialization (for Bayesian net):
prm.log_var_init = {'mean': -10, 'std': 0.1} # The initial value for the log-var parameter (rho) of each weight

# Number of Monte-Carlo iterations (for re-parametrization trick):
prm.n_MC = 1

#  Define optimizer:
prm.optim_func, prm.optim_args = optim.Adam,  {'lr': prm.lr} #'weight_decay': 1e-4
# prm.optim_func, prm.optim_args = optim.SGD, {'lr': prm.lr, 'momentum': 0.9}

# Learning rate decay schedule:
# prm.lr_schedule = {'decay_factor': 0.1, 'decay_epochs': [50, 150]}
prm.lr_schedule = {}  # No decay

# MPB alg  params:
prm.kappa_prior = 2e3  #  parameter of the hyper-prior regularization
prm.kappa_post = 1e-3  # The STD of the 'noise' added to prior
prm.delta = 0.1  #  maximal probability that the bound does not hold

init_from_prior = True  #  False \ True . In meta-testing -  init posterior from learned prior

# Test type:
prm.test_type = 'MaxPosterior' # 'MaxPosterior' / 'MajorityVote' / 'AvgVote'

# path to save the learned meta-parameters
save_path = os.path.join(prm.result_dir, 'model.pt')

task_generator = Task_Generator(prm)

# -------------------------------------------------------------------------------------------
#  Run Meta-Training
# -------------------------------------------------------------------------------------------

start_time = timeit.default_timer()

if prm.mode == 'MetaTrain':

    n_train_tasks = prm.n_train_tasks
    if n_train_tasks:
        # In this case we generate a finite set of train (observed) task before meta-training.
        # Generate the data sets of the training tasks:
        write_to_log('--- Generating {} training-tasks'.format(n_train_tasks), prm)
        train_data_loaders = task_generator.create_meta_batch(prm, n_train_tasks, meta_split='meta_train')

        # Meta-training to learn prior:
        prior_model = meta_train_Bayes_finite_tasks.run_meta_learning(train_data_loaders, prm)
        # save learned prior:
        save_model_state(prior_model, save_path)
        write_to_log('Trained prior saved in ' + save_path, prm)
    else:
        # In this case we observe new tasks generated from the task-distribution in each meta-iteration.
        write_to_log('---- Infinite train tasks - New training tasks are '
                     'drawn from tasks distribution in each iteration...', prm)

        # Meta-training to learn meta-prior (theta params):
        prior_model = meta_train_Bayes_infinite_tasks.run_meta_learning(task_generator, prm)


elif prm.mode == 'LoadMetaModel':

    # Loads  previously training prior.
    # First, create the model:
    prior_model = get_model(prm)
    # Then load the weights:
    load_model_state(prior_model, prm.load_model_path)
    write_to_log('Pre-trained  prior loaded from ' + prm.load_model_path, prm)
else:
    raise ValueError('Invalid mode')

# -------------------------------------------------------------------------------------------
# Generate the data sets of the test tasks:
# -------------------------------------------------------------------------------------------

n_test_tasks = prm.n_test_tasks

limit_train_samples_in_test_tasks = prm.limit_train_samples_in_test_tasks
if limit_train_samples_in_test_tasks == 0:
    limit_train_samples_in_test_tasks = None

write_to_log('---- Generating {} test-tasks with at most {} training samples'.
             format(n_test_tasks, limit_train_samples_in_test_tasks), prm)


test_tasks_data = task_generator.create_meta_batch(prm, n_test_tasks, meta_split='meta_test',
                                                   limit_train_samples=limit_train_samples_in_test_tasks)
#
# -------------------------------------------------------------------------------------------
#  Run Meta-Testing
# -------------------------------------------------------------------------------
write_to_log('Meta-Testing with transferred prior....', prm)

test_err_vec = np.zeros(n_test_tasks)
for i_task in range(n_test_tasks):
    print('Meta-Testing task {} out of {}...'.format(1+i_task, n_test_tasks))
    task_data = test_tasks_data[i_task]
    test_err_vec[i_task], _ = meta_test_Bayes.run_learning(task_data, prior_model, prm, init_from_prior, verbose=0)


# save result
save_run_data(prm, {'test_err_vec': test_err_vec})

# -------------------------------------------------------------------------------------------
#  Print results
# -------------------------------------------------------------------------------------------
#  Print prior analysis
run_prior_analysis(prior_model)

stop_time = timeit.default_timer()
write_to_log('Total runtime: ' +
             time.strftime("%H hours, %M minutes and %S seconds", time.gmtime(stop_time - start_time)),  prm)

#  Print results
write_to_log('----- Final Results: ', prm)
write_to_log('----- Meta-Testing - Avg test err: {:.3}%, STD: {:.3}%'
             .format(100 * test_err_vec.mean(), 100 * test_err_vec.std()), prm)

# -------------------------------------------------------------------------------------------
#  Compare to standard learning
# -------------------------------------------------------------------------------------------
# from Single_Task import learn_single_standard
# test_err_standard = np.zeros(n_test_tasks)
# for i_task in range(n_test_tasks):
#     print('Standard learning task {} out of {}...'.format(i_task, n_test_tasks))
#     task_data = test_tasks_data[i_task]
#     test_err_standard[i_task], _ = learn_single_standard.run_learning(task_data, prm, verbose=0)
#
# write_to_log('Standard - Avg test err: {:.3}%, STD: {:.3}%'.
#              format(100 * test_err_standard.mean(), 100 * test_err_standard.std()), prm)
