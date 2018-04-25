
from __future__ import absolute_import, division, print_function

import timeit
from Models.stochastic_models import get_model
from Utils import common as cmn
from Utils.Bayes_utils import  run_test_Bayes
from Utils.common import grad_step, get_loss_criterion, write_to_log
from PriorMetaLearning.Get_Objective_MPB import get_objective

# -------------------------------------------------------------------------------------------
#  Learning function
# -------------------------------------------------------------------------------------------


def run_meta_learning(task_generator, prm):

    # -------------------------------------------------------------------------------------------
    #  Setting-up
    # -------------------------------------------------------------------------------------------
    # Unpack parameters:
    optim_func, optim_args, lr_schedule =\
        prm.optim_func, prm.optim_args, prm.lr_schedule

    # Loss criterion
    loss_criterion = get_loss_criterion(prm.loss_type)

    # Create a 'dummy' model to generate the set of parameters of the shared prior:
    prior_model = get_model(prm)

    meta_batch_size = prm.meta_batch_size

    n_meta_iterations = prm.n_meta_train_epochs
    n_inner_steps = prm.n_inner_steps

    # -----------------------------------------------------------------------------------------------------------#
    # Main script
    # -----------------------------------------------------------------------------------------------------------#

    # Update Log file
    write_to_log(cmn.get_model_string(prior_model), prm)
    write_to_log('---- Meta-Training with infinite tasks...', prm)

    # -------------------------------------------------------------------------------------------
    #  Run epochs
    # -------------------------------------------------------------------------------------------
    start_time = timeit.default_timer()

    # Training loop:
    test_acc_avg = 0.0
    for i_iter in range(n_meta_iterations):
        prior_model, posteriors_models, test_acc_avg = run_meta_iteration(i_iter, prior_model, task_generator, prm)

    # Note: test_acc_avg is the last checked test error in a meta-training batch
    #  (not the final evaluation which is done on the meta-test tasks)

    stop_time = timeit.default_timer()


    # Update Log file:
    cmn.write_final_result(test_acc_avg, stop_time - start_time, prm, result_name=prm.test_type)

    # Return learned prior:
    return prior_model

# -------------------------------------------------------------------------------------------
#  Training epoch  function
# -------------------------------------------------------------------------------------------
def run_meta_iteration(i_iter, prior_model, task_generator, prm):
    # In each meta-iteration we draw a meta-batch of several tasks
    # Then we take a grad step with prior.

    # Unpack parameters:
    optim_func, optim_args, lr_schedule = \
        prm.optim_func, prm.optim_args, prm.lr_schedule

    # Loss criterion
    loss_criterion = get_loss_criterion(prm.loss_type)
    meta_batch_size = prm.meta_batch_size
    n_inner_steps =  prm.n_inner_steps
    n_meta_iterations = prm.n_meta_train_epochs

    # Generate the data sets of the training-tasks for meta-batch:
    mb_data_loaders = task_generator.create_meta_batch(prm, meta_batch_size, meta_split='meta_train')

    # For each task, prepare an iterator to generate training batches:
    mb_iterators = [iter(mb_data_loaders[ii]['train']) for ii in range(meta_batch_size)]

    # The posteriors models will adjust to new tasks in eacxh meta-batch
    # Create posterior models for each task:
    posteriors_models = [get_model(prm) for _ in range(meta_batch_size)]
    init_from_prior = True
    if init_from_prior:
        for post_model in posteriors_models:
            post_model.load_state_dict(prior_model.state_dict())



    # Gather all tasks posterior params:
    all_post_param = sum([list(posterior_model.parameters()) for posterior_model in posteriors_models], [])

    # Create optimizer for all parameters (posteriors + prior)
    prior_params = list(prior_model.parameters())
    all_params = all_post_param + prior_params
    all_optimizer = optim_func(all_params, **optim_args)
    # all_optimizer = optim_func(prior_params, **optim_args) ## DeBUG


    test_acc_avg = 0.0
    for i_inner_step in range(n_inner_steps):
        # Get objective based on tasks in meta-batch:
        total_objective, info = get_objective(prior_model, prm, mb_data_loaders, mb_iterators,
                                              posteriors_models, loss_criterion, prm.n_train_tasks)

        # Take gradient step with the meta-parameters (theta) based on validation data:
        grad_step(total_objective, all_optimizer, lr_schedule, prm.lr, i_iter)

        # Print status:
        log_interval = 20
        if (i_inner_step) % log_interval == 0:
            batch_acc = info['correct_count'] / info['sample_count']
            print(cmn.status_string(i_iter, n_meta_iterations, i_inner_step, n_inner_steps, batch_acc, total_objective.data[0]) +
                  ' Empiric-Loss: {:.4}\t Task-Comp. {:.4}\t'.
                  format(info['avg_empirical_loss'], info['avg_intra_task_comp']))

    # Print status = on test set of meta-batch:
    log_interval_eval = 10
    if (i_iter) % log_interval_eval == 0 and i_iter > 0:
        test_acc_avg = run_test(mb_data_loaders, posteriors_models, loss_criterion, prm)
        print('Meta-iter: {} \t Meta-Batch Test Acc: {:1.3}\t'.format(i_iter, test_acc_avg))
    # End of inner steps
    return prior_model, posteriors_models, test_acc_avg
# End of meta-iteration function

# -------------------------------------------------------------------------------------------
#  Test evaluation function -
# Evaluate the mean loss on samples from the test sets of the training tasks
# --------------------------------------------------------------------------------------------

def run_test(mb_data_loaders, mb_posteriors_models, loss_criterion, prm):
    n_tasks = len(mb_data_loaders)
    test_acc_avg = 0.0
    n_tests = 0
    for i_task in range(n_tasks):
        model = mb_posteriors_models[i_task]
        test_loader = mb_data_loaders[i_task]['test']
        if len(test_loader) > 0:
            test_acc, test_loss = run_test_Bayes(model, test_loader, loss_criterion, prm, verbose=0)
            n_tests += 1
            test_acc_avg += test_acc

            n_test_samples = len(test_loader.dataset)

            # write_result(
            #     'Train Task {}, Test set: {} -  Average loss: {:.4}, Accuracy: {:.3} of {} samples\n'.format(
            #         prm.test_type, i_task, test_loss, test_acc, n_test_samples), prm)
        else:
            print('Train Task {}, Test set: {} - No test data'.format(i_task, prm.test_type))

    if n_tests > 0:
        test_acc_avg /= n_tests
    return test_acc_avg
