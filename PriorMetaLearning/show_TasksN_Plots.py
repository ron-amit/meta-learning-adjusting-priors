from __future__ import absolute_import, division, print_function
import pickle, os
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

results_dir_names = ['PermutedLabels_TasksN', 'ShuffledPixels100_TasksN', 'ShuffledPixels200_TasksN', 'ShuffledPixels300_TasksN']
root_saved_dir = 'saved/'
paths_to_result_files = [root_saved_dir + name + '/runs_analysis.pkl' for name in results_dir_names]

legend_names = ['Permuted Labels', 'Permuted Pixels - 100 pixel swaps', 'Permuted Pixels - 200 pixel swaps', 'Permuted Pixels - 300 pixel swaps']

n_expirements = len(paths_to_result_files)

plt.figure()

for i_exp in range(n_expirements):

    with open(paths_to_result_files[i_exp], 'rb') as f:
        mean_error_per_tasks_n, std_error_per_tasks_n, n_tasks_vec = pickle.load(f)

    plt.errorbar(n_tasks_vec, 100 * mean_error_per_tasks_n, yerr=100 * std_error_per_tasks_n,
                 label=legend_names[i_exp])
    plt.xticks(n_tasks_vec)


plt.legend()
plt.xlabel('Number of training-tasks', fontsize=18)
plt.ylabel('Error on new task [%]', fontsize=18)

plt.savefig(root_saved_dir + 'TasksN_plot.pdf', format='pdf', bbox_inches='tight')

plt.ylim(0,20) # zoom
plt.xlim(3,10)
plt.savefig(root_saved_dir + 'TasksN_plot_zoom.pdf', format='pdf', bbox_inches='tight')

plt.show()
