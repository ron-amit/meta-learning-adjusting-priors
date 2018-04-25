
from __future__ import absolute_import, division, print_function

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

import torch
from torch.autograd import Variable
import torch.optim as optim


def learn(data_set, complexity_type):

    n_tasks = len(data_set)
    n_dim = data_set[0].shape[1]
    n_samples_list = [task_data.shape[0] for task_data in data_set]

    # Define prior:
    w_P_mu = Variable(torch.randn(n_dim).cuda(), requires_grad=True)
    w_P_log_var = Variable(torch.randn(n_dim).cuda(), requires_grad=True)

    # Init posteriors:
    w_post = Variable(torch.randn(n_tasks, n_dim).cuda(), requires_grad=True)

    learning_rate = 1e-1

    # create your optimizer
    optimizer = optim.Adam([w_post, w_P_mu, w_P_log_var], lr=learning_rate)

    n_epochs = 800
    batch_size = 128

    for i_epoch in range(n_epochs):

        # Sample data batch:
        b_task = np.random.randint(0, n_tasks)  # sample a random task index
        batch_size_curr = min(n_samples_list[b_task], batch_size)
        batch_inds = np.random.choice(n_samples_list[b_task], batch_size_curr, replace=False)
        task_data = torch.from_numpy(data_set[b_task][batch_inds])
        task_data = Variable(task_data.cuda(), requires_grad=False)

        # Empirical Loss:
        w_task = w_post[b_task] # The posterior corresponding to the task in the batch
        empirical_loss = (w_task - task_data).pow(2).mean() # mean over samples and over dimensions

        # Complexity terms:
        sigma_sqr_prior = torch.exp(w_P_log_var)
        complex_term_sum = 0
        for i_task in range(n_tasks):
            neg_log_pdf = 0.5 * torch.sum(w_P_log_var + (w_post[i_task] - w_P_mu).pow(2) / (2*sigma_sqr_prior ))
            n_samples = n_samples_list[i_task]

            if complexity_type == 'Variational_Bayes':
                complex_term_sum += (1 / n_samples) * neg_log_pdf
            elif complexity_type == 'PAC_Bayes':
                complex_term_sum += torch.sqrt((1 / n_samples) * (neg_log_pdf + 100))
            else:
                raise ValueError('Invalid complexity_type')


        hyper_prior_factor =  1e-5 * np.sqrt(1 / n_tasks)
        hyper_prior = torch.sum(sigma_sqr_prior + w_P_mu.pow(2)) * hyper_prior_factor
        # hyper_prior = torch.sum(sigma_sqr_prior) * hyper_prior_factor

        # Total objective:
        complex_term = (1 / n_tasks) * complex_term_sum
        total_objective = empirical_loss + complex_term + hyper_prior

        # Gradient step:
        optimizer.zero_grad()  # zero the gradient buffers
        total_objective.backward()
        optimizer.step()  # Does the update

        if i_epoch % 100 == 0:
            print('Step: {0}, objective: {1}'.format(i_epoch, total_objective.data[0]))

    # Switch  back to numpy:
    w_post = w_post.data.cpu().numpy()
    w_P_mu = w_P_mu.data.cpu().numpy()
    w_P_log_var = w_P_log_var.data.cpu().numpy()
    w_P_sigma = np.exp(0.5 * w_P_log_var)

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
                 label='Task {0}'.format(i_task))
        # plot posterior:
        plt.plot(w_post[i_task][0], w_post[i_task][1], 'o', label='posterior weights {0}'.format(i_task))

    plt.plot(0, 0, 'x', label='hyper-prior ')

    plt.legend()
