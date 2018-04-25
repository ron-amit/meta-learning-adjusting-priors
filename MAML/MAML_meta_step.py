

#
# the code is inspired by: https://github.com/katerakelly/pytorch-maml


from __future__ import absolute_import, division, print_function

import timeit
import random
import numpy as np
from collections import OrderedDict

import torch
from torch.autograd import Variable
from Models.deterministic_models import get_model
from Utils import common as cmn, data_gen
from Utils.common import grad_step, net_norm, correct_rate, get_loss_criterion, write_to_log, count_correct

def meta_step(prm, model, mb_data_loaders, mb_iterators, loss_criterion):

    total_objective = 0
    correct_count = 0
    sample_count = 0

    n_tasks_in_mb = len(mb_data_loaders)

    # ----------- loop over tasks in meta-batch -----------------------------------#
    for i_task in range(n_tasks_in_mb):

        fast_weights = OrderedDict((name, param) for (name, param) in model.named_parameters())

        # ----------- gradient steps loop -----------------------------------#
        for i_step in range(prm.n_meta_train_grad_steps):

            # get batch variables:
            batch_data = data_gen.get_next_batch_cyclic(mb_iterators[i_task],
                                                        mb_data_loaders[i_task]['train'])
            inputs, targets = data_gen.get_batch_vars(batch_data, prm)

            # Debug
            # print(targets[0].data[0])  # print first image label
            # import matplotlib.pyplot as plt
            # plt.imshow(inputs[0].cpu().data[0].numpy())  # show first image
            # plt.show()

            if i_step == 0:
                outputs = model(inputs)
            else:
                outputs = model(inputs, fast_weights)
            # Empirical Loss on current task:
            task_loss = loss_criterion(outputs, targets)
            grads = torch.autograd.grad(task_loss, fast_weights.values(), create_graph=True)

            fast_weights = OrderedDict((name, param - prm.alpha * grad)
                                       for ((name, param), grad) in zip(fast_weights.items(), grads))
        # end grad steps loop

        # Sample new  (validation) data batch for this task:
        if hasattr(prm, 'MAML_Use_Test_Data') and prm.MAML_Use_Test_Data:
            batch_data = data_gen.get_next_batch_cyclic(mb_iterators[i_task],
                                                        mb_data_loaders[i_task]['test'])
        else:
            batch_data = data_gen.get_next_batch_cyclic(mb_iterators[i_task],
                                                     mb_data_loaders[i_task]['train'])


        inputs, targets = data_gen.get_batch_vars(batch_data, prm)
        outputs = model(inputs, fast_weights)
        total_objective += loss_criterion(outputs, targets)
        correct_count += count_correct(outputs, targets)
        sample_count += inputs.size(0)
    # end loop over tasks in  meta-batch

    info = {'sample_count': sample_count, 'correct_count': correct_count}
    return total_objective, info
