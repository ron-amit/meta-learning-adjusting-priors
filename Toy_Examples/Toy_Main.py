from __future__ import absolute_import, division, print_function

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from Utils.common import  set_random_seed

matplotlib.rcParams.update({'font.size': 10})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
# -------------------------------------------------------------------------------------------
#  Create data
# -------------------------------------------------------------------------------------------

# Random seed:
seed = 2
if not seed == 0:
    set_random_seed(seed)


# -------------------------------------------------------------------------------------------
# Define scenario
# -------------------------------------------------------------------------------------------

n_dim = 2

data_type = 1 # 0 \ 1

if data_type == 0:
    n_tasks = 2
    # number of samples in each task:
    n_samples_list =[10, 200]
    # True means vector for each task [n_dim x n_tasks]:
    true_mu = [[-1.0,-1.0], [+1.0, +1.0]]
    # True sigma vector for each task [n_dim x n_tasks]:
    true_sigma = [[0.1,0.1], [0.1, 0.1]]

elif data_type == 1:
    n_tasks = 2
    # number of samples in each task:
    n_samples_list = [100, 100]
    # True means vector for each task [n_dim x n_tasks]:
    true_mu = [[2, 1], [4, 1]]
    # True sigma vector for each task [n_dim x n_tasks]:
    true_sigma = [[0.1, 0.1], [0.1, 0.1]]

else:
    raise ValueError('Invalid data_type')


# -------------------------------------------------------------------------------------------
#  Generate data samples
# -------------------------------------------------------------------------------------------
data_set = []
for i_task in range(n_tasks):
    task_data = np.random.multivariate_normal(
        mean=true_mu[i_task],
        cov=np.diag(true_sigma[i_task]),
        size=n_samples_list[i_task]).astype(np.float32)

    data_set.append(task_data)

# -------------------------------------------------------------------------------------------
#  Learning
# -------------------------------------------------------------------------------------------
learning_type = 'MetaLearnPosteriors' # 'Standard' \ 'Bayes_FixedPrior' \ 'MetaLearnPosteriors' \ MetaLearnWeights
# 'Standard' = Learn optimal weights in each task separately
# 'Bayes_FixedPrior' = Learn posteriors for each task, assuming a fixed shared prior
# 'MetaLearnPosteriors' = Learn weights for each task and the shared prior jointly
#

if learning_type == 'Standard':
    import toy_standard
    toy_standard.learn(data_set)

if learning_type == 'Bayes_FixedPrior':
    import toy_Bayes_FixedPrior
    toy_Bayes_FixedPrior.learn(data_set)

if learning_type == 'MetaLearnPosteriors':
    import toy_MetaLearnPosteriors
    complexity_type = 'PAC_Bayes_McAllaster' # 'PAC_Bayes_McAllaster' \ 'Variational_Bayes' \ 'KL'
    toy_MetaLearnPosteriors.learn(data_set, complexity_type)

if learning_type == 'MetaLearnWeights':
    import toy_MetaLearnWeights
    complexity_type = 'PAC_Bayes'  # 'Variational_Bayes' / 'PAC_Bayes' /
    toy_MetaLearnWeights.learn(data_set, complexity_type)


plt.savefig('ToyFig1.pdf', format='pdf', bbox_inches='tight')

plt.show()

