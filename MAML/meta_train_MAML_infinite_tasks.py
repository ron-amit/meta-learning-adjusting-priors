
#
# the code is inspired by: https://github.com/katerakelly/pytorch-maml


from __future__ import absolute_import, division, print_function

import timeit

from Models.deterministic_models import get_model
from Utils import common as cmn, data_gen
from Utils.common import grad_step, get_loss_criterion, write_to_log
from MAML.MAML_meta_step import meta_step
# -------------------------------------------------------------------------------------------
#  Learning function
# -------------------------------------------------------------------------------------------
def run_meta_learning(prm, task_generator):

    # -------------------------------------------------------------------------------------------
    #  Setting-up
    # -------------------------------------------------------------------------------------------
    # Unpack parameters:
    optim_func, optim_args, lr_schedule =\
        prm.optim_func, prm.optim_args, prm.lr_schedule

    n_iterations = prm.n_meta_train_iterations

    # Loss criterion
    loss_criterion = get_loss_criterion(prm.loss_type)

    # Create a 'dummy' model to generate the set of parameters of the shared initial point (theta):
    model = get_model(prm)
    model.train()

    # Create optimizer for meta-params (theta)
    meta_params = list(model.parameters())

    meta_optimizer = optim_func(meta_params, **optim_args)

    meta_batch_size = prm.meta_batch_size
    # -------------------------------------------------------------------------------------------
    #  Training epoch  function
    # -------------------------------------------------------------------------------------------
    def run_meta_iteration(i_iter):
        # In each meta-iteration we draw a meta-batch of several tasks
        # Then we take a grad step with theta.

        # Generate the data sets of the training-tasks for meta-batch:
        mb_data_loaders = task_generator.create_meta_batch(prm, meta_batch_size, meta_split='meta_train')

        # For each task, prepare an iterator to generate training batches:
        mb_iterators = [iter(mb_data_loaders[ii]['train']) for ii in range(meta_batch_size)]

        # Get objective based on tasks in meta-batch:
        total_objective, info = meta_step(prm, model, mb_data_loaders, mb_iterators, loss_criterion)

        # Take gradient step with the meta-parameters (theta) based on validation data:
        grad_step(total_objective, meta_optimizer, lr_schedule, prm.lr, i_iter)

        # Print status:
        log_interval = 5
        if (i_iter) % log_interval == 0:
            batch_acc = info['correct_count'] / info['sample_count']
            print(cmn.status_string(i_iter, n_iterations, 1, 1, batch_acc, total_objective.data[0]))


    # end run_meta_iteration()

    # -----------------------------------------------------------------------------------------------------------#
    # Main script
    # -----------------------------------------------------------------------------------------------------------#

    # Update Log file
    write_to_log(cmn.get_model_string(model), prm)
    write_to_log('---- Meta-Training with infinite tasks...', prm)

    # -------------------------------------------------------------------------------------------
    #  Run epochs
    # -------------------------------------------------------------------------------------------
    start_time = timeit.default_timer()

    # Training loop:
    for i_iter in range(n_iterations):
        run_meta_iteration(i_iter)

    stop_time = timeit.default_timer()

    # Update Log file:
    cmn.write_final_result(0.0, stop_time - start_time, prm)

    # Return learned meta-parameters:
    return model


