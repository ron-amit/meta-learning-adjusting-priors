from __future__ import absolute_import, division, print_function

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Ellipse

import torch
from torch.autograd import Variable
import torch.optim as optim


# -------------------------------------------------------------------------------------------
#  Standard Learning
# -------------------------------------------------------------------------------------------
def learn(data_set):

    n_tasks = len(data_set)
    n_dim = data_set[0].shape[1]
    n_samples_list = [task_data.shape[0] for task_data in data_set]

    # Init weights:
    w = Variable(torch.randn(n_tasks, n_dim).cuda(), requires_grad=True)

    learning_rate = 1e-1

    # create your optimizer
    optimizer = optim.Adam([w], lr=learning_rate)

    n_epochs = 300
    batch_size = 128

    for i_epoch in range(n_epochs):

        # Sample data batch:
        b_task = np.random.randint(0, n_tasks)  # sample a random task index
        batch_size_curr = min(n_samples_list[b_task], batch_size)
        batch_inds = np.random.choice(n_samples_list[b_task], batch_size_curr, replace=False)
        task_data = torch.from_numpy(data_set[b_task][batch_inds])
        task_data = Variable(task_data.cuda(), requires_grad=False)

        # Loss:
        loss = (w[b_task] - task_data).pow(2).mean()

        # Gradient step:
        optimizer.zero_grad()  # zero the gradient buffers
        loss.backward()
        optimizer.step()  # Does the update

        if i_epoch % 100 == 0:
            print('Step: {0}, loss: {1}'.format(i_epoch, loss.data[0]))

    # Switch learned parameters back to numpy:
    w = w.data.cpu().numpy()

    #  Plots:
    fig1 = plt.figure()
    ax = plt.subplot(0, aspect='equal')
    for i_task in range(n_tasks):
        plt.plot(data_set[i_task][:, 0], data_set[i_task][:, 1], '.',
                 label='Task {0}'.format(i_task))
        plt.plot(w[i_task][0], w[i_task][1], 'x', label='Learned w in task {0}'.format(i_task))
    plt.legend()

