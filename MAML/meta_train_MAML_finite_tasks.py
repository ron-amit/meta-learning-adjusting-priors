
#
# the code is inspired by: https://github.com/katerakelly/pytorch-maml


from __future__ import absolute_import, division, print_function

import timeit
import random
import numpy as np

from Models.deterministic_models import get_model
from Utils import common as cmn
from Utils.common import grad_step, get_loss_criterion, write_to_log, get_value
from MAML.MAML_meta_step import meta_step
# -------------------------------------------------------------------------------------------
#  Learning function
# -------------------------------------------------------------------------------------------
def run_meta_learning(train_data_loaders, prm):

    # -------------------------------------------------------------------------------------------
    #  Setting-up
    # -------------------------------------------------------------------------------------------
    # Unpack parameters:
    optim_func, optim_args, lr_schedule =\
        prm.optim_func, prm.optim_args, prm.lr_schedule

    # Loss criterion
    loss_criterion = get_loss_criterion(prm.loss_type)

    n_tasks = len(train_data_loaders)

    # Create a 'dummy' model to generate the set of parameters of the shared initial point (theta):
    model = get_model(prm)
    model.train()

    # Create optimizer for meta-params (theta)
    meta_params = list(model.parameters())

    meta_optimizer = optim_func(meta_params, **optim_args)

    # number of sample-batches in each task:
    n_batch_list = [len(data_loader['train']) for data_loader in train_data_loaders]

    n_batches_per_task = np.max(n_batch_list)
    # note: if some tasks have less data that other tasks - it may be sampled more than once in an epoch

    # -------------------------------------------------------------------------------------------
    #  Training epoch  function
    # -------------------------------------------------------------------------------------------
    def run_train_epoch(i_epoch):

        # For each task, prepare an iterator to generate training batches:
        train_iterators = [iter(train_data_loaders[ii]['train']) for ii in range(n_tasks)]

        # The task order to take batches from:
        task_order = []
        task_ids_list = list(range(n_tasks))
        for i_batch in range(n_batches_per_task):
            random.shuffle(task_ids_list)
            task_order += task_ids_list

        # each meta-batch includes several tasks
        # we take a grad step with theta after each meta-batch
        meta_batch_starts = list(range(0, len(task_order), prm.meta_batch_size))
        n_meta_batches = len(meta_batch_starts)

        # ----------- meta-batches loop (batches of tasks) -----------------------------------#
        for i_meta_batch in range(n_meta_batches):

            meta_batch_start = meta_batch_starts[i_meta_batch]
            task_ids_in_meta_batch = task_order[meta_batch_start: (meta_batch_start + prm.meta_batch_size)]
            n_tasks_in_batch = len(task_ids_in_meta_batch)  # it may be less than  prm.meta_batch_size at the last one
            # note: it is OK if some task appear several times in the meta-batch

            mb_data_loaders = [train_data_loaders[task_id] for task_id in task_ids_in_meta_batch]
            mb_iterators = [train_iterators[task_id] for task_id in task_ids_in_meta_batch]

            # Get objective based on tasks in meta-batch:
            total_objective, info = meta_step(prm, model, mb_data_loaders, mb_iterators, loss_criterion)

            # Take gradient step with the meta-parameters (theta) based on validation data:
            grad_step(total_objective, meta_optimizer, lr_schedule, prm.lr, i_epoch)

            # Print status:
            log_interval = 200
            if i_meta_batch % log_interval == 0:
                batch_acc = info['correct_count'] / info['sample_count']
                print(cmn.status_string(i_epoch, num_epochs, i_meta_batch, n_meta_batches, batch_acc, get_value(total_objective)))
        # end  meta-batches loop

    # end run_epoch()

    # -----------------------------------------------------------------------------------------------------------#
    # Main script
    # -----------------------------------------------------------------------------------------------------------#

    # Update Log file
    write_to_log(cmn.get_model_string(model), prm)
    write_to_log('---- Meta-Training set: {0} tasks'.format(len(train_data_loaders)), prm)

    # -------------------------------------------------------------------------------------------
    #  Run epochs
    # -------------------------------------------------------------------------------------------
    start_time = timeit.default_timer()

    num_epochs = int(np.ceil(prm.n_meta_train_iterations / np.ceil(n_tasks / prm.meta_batch_size)))

    # Training loop:
    for i_epoch in range(num_epochs):
        run_train_epoch(i_epoch)

    stop_time = timeit.default_timer()

    # Update Log file:
    cmn.write_final_result(0.0, stop_time - start_time, prm)

    # Return learned meta-parameters:
    return model


