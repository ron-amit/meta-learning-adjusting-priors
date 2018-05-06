
from __future__ import absolute_import, division, print_function

import timeit
import random
import numpy as np
from Models.stochastic_models import get_model
from Utils import common as cmn
from Utils.Bayes_utils import run_test_Bayes
from Utils.common import grad_step, get_loss_criterion, write_to_log, get_value
from PriorMetaLearning.Get_Objective_MPB import get_objective

# -------------------------------------------------------------------------------------------
#  Learning function
# -------------------------------------------------------------------------------------------


def run_meta_learning(data_loaders, prm):

    # -------------------------------------------------------------------------------------------
    #  Setting-up
    # -------------------------------------------------------------------------------------------
    # Unpack parameters:
    optim_func, optim_args, lr_schedule =\
        prm.optim_func, prm.optim_args, prm.lr_schedule

    # Loss criterion
    loss_criterion = get_loss_criterion(prm.loss_type)

    n_train_tasks = len(data_loaders)

    # Create posterior models for each task:
    posteriors_models = [get_model(prm) for _ in range(n_train_tasks)]

    # Create a 'dummy' model to generate the set of parameters of the shared prior:
    prior_model = get_model(prm)

    # Gather all tasks posterior params:
    all_post_param = sum([list(posterior_model.parameters()) for posterior_model in posteriors_models], [])

    # Create optimizer for all parameters (posteriors + prior)
    prior_params = list(prior_model.parameters())
    all_params = all_post_param + prior_params
    all_optimizer = optim_func(all_params, **optim_args)

    # number of sample-batches in each task:
    n_batch_list = [len(data_loader['train']) for data_loader in data_loaders]

    n_batches_per_task = np.max(n_batch_list)

    # -------------------------------------------------------------------------------------------
    #  Training epoch  function
    # -------------------------------------------------------------------------------------------
    def run_train_epoch(i_epoch):

        # For each task, prepare an iterator to generate training batches:
        train_iterators = [iter(data_loaders[ii]['train']) for ii in range(n_train_tasks)]

        # The task order to take batches from:
        # The meta-batch will be balanced - i.e, each task will appear roughly the same number of times
        # note: if some tasks have less data that other tasks - it may be sampled more than once in an epoch
        task_order = []
        task_ids_list = list(range(n_train_tasks))
        for i_batch in range(n_batches_per_task):
            random.shuffle(task_ids_list)
            task_order += task_ids_list
        # Note: this method ensures each training sample in each task is drawn in each epoch.
        # If all the tasks have the same number of sample, then each sample is drawn exactly once in an epoch.

        # ----------- meta-batches loop (batches of tasks) -----------------------------------#
        # each meta-batch includes several tasks
        # we take a grad step with theta after each meta-batch
        meta_batch_starts = list(range(0, len(task_order), prm.meta_batch_size))
        n_meta_batches = len(meta_batch_starts)

        for i_meta_batch in range(n_meta_batches):


            meta_batch_start = meta_batch_starts[i_meta_batch]
            task_ids_in_meta_batch = task_order[meta_batch_start: (meta_batch_start + prm.meta_batch_size)]
            # meta-batch size may be less than  prm.meta_batch_size at the last one
            # note: it is OK if some tasks appear several times in the meta-batch

            mb_data_loaders = [data_loaders[task_id] for task_id in task_ids_in_meta_batch]
            mb_iterators = [train_iterators[task_id] for task_id in task_ids_in_meta_batch]
            mb_posteriors_models = [posteriors_models[task_id] for task_id in task_ids_in_meta_batch]

            # Get objective based on tasks in meta-batch:
            total_objective, info = get_objective(prior_model, prm, mb_data_loaders,
                                                  mb_iterators, mb_posteriors_models, loss_criterion, n_train_tasks)

            # Take gradient step with the shared prior and all tasks' posteriors:
            grad_step(total_objective, all_optimizer, lr_schedule, prm.lr, i_epoch)

            # Print status:
            log_interval = 200
            if i_meta_batch % log_interval == 0:
                batch_acc = info['correct_count'] / info['sample_count']
                print(cmn.status_string(i_epoch,  prm.n_meta_train_epochs, i_meta_batch, n_meta_batches, batch_acc, get_value(total_objective)) +
                      ' Empiric-Loss: {:.4}\t Task-Comp. {:.4}\t Meta-Comp.: {:.4}'.
                      format(info['avg_empirical_loss'], info['avg_intra_task_comp'], info['meta_comp']))
        # end  meta-batches loop

    # end run_epoch()

    # -------------------------------------------------------------------------------------------
    #  Test evaluation function -
    # Evaluate the mean loss on samples from the test sets of the training tasks
    # --------------------------------------------------------------------------------------------
    def run_test():
        test_acc_avg = 0.0
        n_tests = 0
        for i_task in range(n_train_tasks):
            model = posteriors_models[i_task]
            test_loader = data_loaders[i_task]['test']
            if len(test_loader) > 0:
                test_acc, test_loss = run_test_Bayes(model, test_loader, loss_criterion, prm)
                n_tests += 1
                test_acc_avg += test_acc

                n_test_samples = len(test_loader.dataset)

                write_to_log('Train Task {}, Test set: {} -  Average loss: {:.4}, Accuracy: {:.3} (of {} samples)\n'.format(
                    i_task, prm.test_type, test_loss, test_acc, n_test_samples), prm)
            else:
                print('Train Task {}, Test set: {} - No test data'.format(i_task, prm.test_type))

        if n_tests > 0:
            test_acc_avg /= n_tests
        return test_acc_avg

    # -----------------------------------------------------------------------------------------------------------#
    # Main script
    # -----------------------------------------------------------------------------------------------------------#

    # Update Log file

    write_to_log(cmn.get_model_string(prior_model), prm)
    write_to_log('---- Meta-Training set: {0} tasks'.format(len(data_loaders)), prm)

    # -------------------------------------------------------------------------------------------
    #  Run epochs
    # -------------------------------------------------------------------------------------------
    start_time = timeit.default_timer()

    # Training loop:
    for i_epoch in range(prm.n_meta_train_epochs):
        run_train_epoch(i_epoch)

    stop_time = timeit.default_timer()

    # Test:
    test_acc_avg = run_test()

    # Update Log file:
    cmn.write_final_result(test_acc_avg, stop_time - start_time, prm, result_name=prm.test_type)

    # Return learned prior:
    return prior_model


