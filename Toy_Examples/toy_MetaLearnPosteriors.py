
from __future__ import absolute_import, division, print_function

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse
from Utils.common import get_value
import torch
from torch.autograd import Variable
import torch.optim as optim


def learn(data_set, complexity_type):

    n_tasks = len(data_set)
    n_dim = data_set[0].shape[1]
    n_samples_list = [task_data.shape[0] for task_data in data_set]

    # Define prior:
    w_P_mu = Variable(torch.randn(n_dim).cuda(), requires_grad=True)
    w_P_log_sigma = Variable(torch.randn(n_dim).cuda(), requires_grad=True)

    # Init posteriors:
    w_mu = Variable(torch.randn(n_tasks, n_dim).cuda(), requires_grad=True)
    w_log_sigma = Variable(torch.randn(n_tasks, n_dim).cuda(), requires_grad=True)

    learning_rate = 1e-1

    # create your optimizer
    optimizer = optim.Adam([w_mu, w_log_sigma, w_P_mu, w_P_log_sigma], lr=learning_rate)

    n_epochs = 800
    batch_size = 128

    for i_epoch in range(n_epochs):

        # Sample data batch:
        b_task = np.random.randint(0, n_tasks)  # sample a random task index
        batch_size_curr = min(n_samples_list[b_task], batch_size)
        batch_inds = np.random.choice(n_samples_list[b_task], batch_size_curr, replace=False)
        task_data = torch.from_numpy(data_set[b_task][batch_inds])
        task_data = Variable(task_data.cuda(), requires_grad=False)

        # Re-Parametrization:
        w_sigma = torch.exp(w_log_sigma[b_task])
        epsilon = Variable(torch.randn(n_dim).cuda(), requires_grad=False)
        w = w_mu[b_task] + w_sigma * epsilon

        # Empirical Loss:
        empirical_loss = (w - task_data).pow(2).mean()

        # Complexity terms:
        sigma_sqr_prior = torch.exp(2 * w_P_log_sigma)
        complex_term_sum = 0
        for i_task in range(n_tasks):

            sigma_sqr_post = torch.exp(2 * w_log_sigma[i_task])

            kl_dist = torch.sum(w_P_log_sigma - w_log_sigma[i_task] +
                                ((w_mu[i_task] - w_P_mu).pow(2) + sigma_sqr_post) /
                                (2 * sigma_sqr_prior) - 0.5)

            n_samples = n_samples_list[i_task]

            if complexity_type == 'PAC_Bayes_McAllaster':
                delta = 0.95
                complex_term_sum += torch.sqrt((1 / (2 * n_samples)) *
                                           (kl_dist + np.log(2 * np.sqrt(n_samples) / delta)))

            elif complexity_type == 'Variational_Bayes':
                complex_term_sum += (1 / n_samples) * kl_dist

            elif complexity_type == 'KL':
                complex_term_sum += kl_dist
            else:
                raise ValueError('Invalid complexity_type')


        hyper_prior_factor =  1e-6 * np.sqrt(1 / n_tasks)
        hyper_prior = torch.sum(sigma_sqr_prior + w_P_mu.pow(2))  * hyper_prior_factor

        # Total objective:
        complex_term = (1 / n_tasks) * complex_term_sum
        objective = empirical_loss + complex_term + hyper_prior

        # Gradient step:
        optimizer.zero_grad()  # zero the gradient buffers
        objective.backward()
        optimizer.step()  # Does the update

        if i_epoch % 100 == 0:
            print('Step: {0}, objective: {1}'.format(i_epoch, get_value(objective)))

    # Switch  back to numpy:
    w_mu = w_mu.data.cpu().numpy()
    w_log_sigma = w_log_sigma.data.cpu().numpy()
    w_sigma = np.exp(w_log_sigma)
    w_P_mu = w_P_mu.data.cpu().numpy()
    w_P_log_sigma = w_P_log_sigma.data.cpu().numpy()
    w_P_sigma = np.exp(w_P_log_sigma)

    #  Plots:
    fig1 = plt.figure()
    ax = plt.subplot(111, aspect='equal')
    # plot prior:
    plt.plot(w_P_mu[0], w_P_mu[1], 'o', label='prior mean ')
    ell = Ellipse(xy=(w_P_mu[0], w_P_mu[1]),
                  width=w_P_sigma[0], height=w_P_sigma[1],
                  angle=0, color='blue')
    ell.set_facecolor('none')
    ax.add_artist(ell)
    for i_task in range(n_tasks):
        # plot task data points:
        plt.plot(data_set[i_task][:, 0], data_set[i_task][:, 1], '.',
                 label='Task {0} samples'.format(1+i_task))
        # plot posterior:
        plt.plot(w_mu[i_task][0], w_mu[i_task][1], 'o', label='posterior {0} mean'.format(1+i_task))
        ell = Ellipse(xy=(w_mu[i_task][0], w_mu[i_task][1]),
                      width=w_sigma[i_task][0], height=w_sigma[i_task][1],
                      angle=0, color='black')
        ell.set_facecolor('none')
        ax.add_artist(ell)

    # plt.plot(0, 0, 'x', label='hyper-prior ')

    # plt.legend(loc='upper left')
    plt.legend()
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    # plt.xticks( np.arange(1, 5) )
    # plt.yticks( np.arange(0, 2, 0.5) )
