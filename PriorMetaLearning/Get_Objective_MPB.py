from __future__ import absolute_import, division, print_function


# from Models.stochastic_models import get_model
from Models.stochastic_models import get_model
from Utils import common as cmn, data_gen
from Utils.Bayes_utils import get_bayes_task_objective, run_test_Bayes, get_meta_complexity_term
from Utils.common import grad_step, net_norm, count_correct, get_loss_criterion, get_value

# -------------------------------------------------------------------------------------------
#
# -------------------------------------------------------------------------------------------
def get_objective(prior_model, prm, mb_data_loaders, mb_iterators, mb_posteriors_models, loss_criterion, n_train_tasks):
    '''  Calculate objective based on tasks in meta-batch '''
    # note: it is OK if some tasks appear several times in the meta-batch

    n_tasks_in_mb = len(mb_data_loaders)

    sum_empirical_loss = 0
    sum_intra_task_comp = 0
    correct_count = 0
    sample_count = 0

    # KLD between hyper-posterior and hyper-prior:
    hyper_kl = (1 / (2 * prm.kappa_prior**2)) * net_norm(prior_model, p=2)

    # Hyper-prior term:
    meta_complex_term = get_meta_complexity_term(hyper_kl, prm, n_train_tasks)

    # ----------- loop over tasks in meta-batch -----------------------------------#
    for i_task in range(n_tasks_in_mb):

        n_samples = mb_data_loaders[i_task]['n_train_samples']

        # get sample-batch data from current task to calculate the empirical loss estimate:
        batch_data = data_gen.get_next_batch_cyclic(mb_iterators[i_task], mb_data_loaders[i_task]['train'])

        # The posterior model corresponding to the task in the batch:
        post_model = mb_posteriors_models[i_task]
        post_model.train()

        # Monte-Carlo iterations:
        n_MC = prm.n_MC
        task_empirical_loss = 0
        task_complexity = 0
        # ----------- Monte-Carlo loop  -----------------------------------#
        for i_MC in range(n_MC):
            # get batch variables:
            inputs, targets = data_gen.get_batch_vars(batch_data, prm)

            # Debug
            # print(targets[0].data[0])  # print first image label
            # import matplotlib.pyplot as plt
            # plt.imshow(inputs[0].cpu().data[0].numpy())  # show first image
            # plt.show()

            # Empirical Loss on current task:
            outputs = post_model(inputs)
            curr_empirical_loss = loss_criterion(outputs, targets)

            correct_count += count_correct(outputs, targets)
            sample_count += inputs.size(0)

            # Intra-task complexity of current task:
            curr_empirical_loss, curr_complexity = get_bayes_task_objective(
                prm, prior_model, post_model,
                n_samples, curr_empirical_loss, hyper_kl, n_train_tasks=n_train_tasks)

            task_empirical_loss += (1 / n_MC) * curr_empirical_loss
            task_complexity += (1 / n_MC) * curr_complexity
        # end Monte-Carlo loop

        sum_empirical_loss += task_empirical_loss
        sum_intra_task_comp += task_complexity

    # end loop over tasks in meta-batch
    avg_empirical_loss = (1 / n_tasks_in_mb) * sum_empirical_loss
    avg_intra_task_comp = (1 / n_tasks_in_mb) * sum_intra_task_comp


    # Approximated total objective:
    total_objective = avg_empirical_loss + avg_intra_task_comp + meta_complex_term

    info = {'sample_count': get_value(sample_count), 'correct_count': get_value(correct_count),
                  'avg_empirical_loss': get_value(avg_empirical_loss),
                  'avg_intra_task_comp': get_value(avg_intra_task_comp),
                  'meta_comp': get_value(meta_complex_term)}
    return total_objective, info
