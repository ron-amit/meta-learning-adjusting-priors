
#
# the code is inspired by: https://github.com/katerakelly/pytorch-maml

from __future__ import absolute_import, division, print_function

import timeit

from Models.deterministic_models import get_model
from Utils import common as cmn, data_gen
from Utils.common import grad_step, correct_rate, get_loss_criterion, write_to_log, count_correct
from torch.optim import SGD

def run_learning(task_data, meta_model, prm, verbose=1):

    # -------------------------------------------------------------------------------------------
    #  Setting-up
    # -------------------------------------------------------------------------------------------

    # Loss criterion
    loss_criterion = get_loss_criterion(prm.loss_type)

    # Create model for task:
    task_model = get_model(prm)
    
    #  Load initial point from meta-parameters:
    task_model.load_state_dict(meta_model.state_dict())

    # The data-sets of the new task:
    train_loader = task_data['train']
    test_loader = task_data['test']
    n_train_samples = len(train_loader.dataset)
    n_batches = len(train_loader)

    #  Get task optimizer:
    task_optimizer = SGD(task_model.parameters(), lr=prm.alpha)
    # In meta-testing, use SGD with step-size alpha

    # -------------------------------------------------------------------------------------------
    #  Learning  function
    # -------------------------------------------------------------------------------------------

    def run_meta_test_learning(task_model, train_loader):      

        task_model.train()
        train_loader_iter = iter(train_loader)

        # Gradient steps (training) loop
        for i_grad_step in range(prm.n_meta_test_grad_steps):
            # get batch:
            batch_data = data_gen.get_next_batch_cyclic(train_loader_iter, train_loader)
            inputs, targets = data_gen.get_batch_vars(batch_data, prm)

            # Calculate empirical loss:
            outputs = task_model(inputs)
            task_objective = loss_criterion(outputs, targets)

            # Take gradient step with the task weights:
            grad_step(task_objective, task_optimizer)

        # end gradient step loop

        return task_model

    # -------------------------------------------------------------------------------------------
    #  Test evaluation function
    # --------------------------------------------------------------------------------------------
    def run_test(model, test_loader):
        model.eval()
        test_loss = 0
        n_correct = 0
        for batch_data in test_loader:
            inputs, targets = data_gen.get_batch_vars(batch_data, prm, is_test=True)
            outputs = model(inputs)
            test_loss += loss_criterion(outputs, targets)  # sum the mean loss in batch
            n_correct += count_correct(outputs, targets)

        n_test_samples = len(test_loader.dataset)
        n_test_batches = len(test_loader)
        test_loss = test_loss.data[0] / n_test_batches
        test_acc = n_correct / n_test_samples
        print('\nTest set: Average loss: {:.4}, Accuracy: {:.3} ( {}/{})\n'.format(
            test_loss, test_acc, n_correct, n_test_samples))
        return test_acc
          

    # -----------------------------------------------------------------------------------------------------------#
    # Update Log file
    if verbose == 1:
        write_to_log('Total number of steps: {}'.format(n_batches * prm.num_epochs), prm)

    # -------------------------------------------------------------------------------------------
    #  Run epochs
    # -------------------------------------------------------------------------------------------
    start_time = timeit.default_timer()

    # Training loop:
    task_model = run_meta_test_learning(task_model, train_loader)

    # Test:
    test_acc = run_test(task_model, test_loader)

    stop_time = timeit.default_timer()
    cmn.write_final_result(test_acc, stop_time - start_time, prm, verbose=verbose)

    test_err = 1 - test_acc
    return test_err, task_model
