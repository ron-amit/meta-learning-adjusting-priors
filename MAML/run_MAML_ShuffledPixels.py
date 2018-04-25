from subprocess import call
import os
import timeit, time
import numpy as np
from Utils.common import load_run_data

# Runs meta-training with the specified hyper-parameters
# and meta-testing with a range of 'meta-test gradient steps'

n_train_tasks = 10
n_pixels_shuffles = 200

alpha = 0.01
n_meta_train_grad_steps = 2
run_learning = True   # If false, just show results

min_test_grad_steps = 1
max_test_grad_steps = 20
n_meta_test_grad_steps_vec = list(range(min_test_grad_steps, 1+max_test_grad_steps))
meta_train_in_this_run = 1  # we can meta-train just once and use the learned meta-parameters to meta-test with different number of gradient steps

base_run_name = 'Shuffled_{}_Pixels_{}_Tasks_Alpha_{}_TrainGrads_{}'.format(n_pixels_shuffles, n_train_tasks, alpha, n_meta_train_grad_steps)
base_run_name = base_run_name.replace('.','_')
sub_runs_names = [base_run_name + '/' + 'TestGrads_' + str(n_meta_test_grad_steps) for n_meta_test_grad_steps in n_meta_test_grad_steps_vec]

root_saved_dir = 'saved/'

start_time = timeit.default_timer()


if run_learning:
    for i_run, n_meta_test_grad_steps in enumerate(n_meta_test_grad_steps_vec):
        print('---------- n_meta_test_grad_steps = {}'.format(n_meta_test_grad_steps))
        if n_meta_test_grad_steps == meta_train_in_this_run:
            mode = 'MetaTrain'
        else:
            mode = 'LoadMetaModel'

        call(['python', 'main_MAML.py',
              '--run-name', sub_runs_names[i_run],
              '--mode', mode,
              '--load_model_path', root_saved_dir + base_run_name + '/' + 'TestGrads_' +  str(meta_train_in_this_run) + '/model.pt',
              '--data-source', 'MNIST',
              '--n_train_tasks', str(n_train_tasks),
              '--data-transform', 'Shuffled_Pixels',
              '--n_pixels_shuffles', str(n_pixels_shuffles),
              '--model-name', 'FcNet3',
              # MAML hyper-parameters:
              '--alpha', str(alpha),
              '--n_meta_train_grad_steps', str(n_meta_train_grad_steps),
              '--n_meta_train_iterations', '300', #  '300',
              '--meta_batch_size', '32',
              '--n_meta_test_grad_steps', str(n_meta_test_grad_steps),
              '--n_test_tasks', '20',  #  '20',
              '--limit_train_samples_in_test_tasks', '2000',
              ])

    stop_time = timeit.default_timer()
    print('Total runtime: ' +
          time.strftime("%H hours, %M minutes and %S seconds", time.gmtime(stop_time - start_time)))

# -------------------------------------------------------------------------------------------
# Analyze the experiments
# -------------------------------------------------------------------------------------------
n_runs = len(n_meta_test_grad_steps_vec)
mean_error_per_run = np.zeros(len(n_meta_test_grad_steps_vec))
std_error_per_run = np.zeros(len(n_meta_test_grad_steps_vec))

for i_run, n_meta_test_grad_steps in enumerate(n_meta_test_grad_steps_vec):
    run_result_path = os.path.join(root_saved_dir, sub_runs_names[i_run])
    prm, info_dict = load_run_data(run_result_path)
    test_err_vec = info_dict['test_err_vec']
    mean_error_per_run[i_run] = test_err_vec.mean()
    std_error_per_run[i_run] = test_err_vec.std()
    print('Meta-Test Grad Steps: {}, Mean Err: {}%, STD: {}%'.format(n_meta_test_grad_steps, 100*mean_error_per_run[i_run], 100*std_error_per_run[i_run]))


# # Saving the analysis:
# with open(os.path.join(root_saved_dir, base_run_name, 'runs_analysis.pkl'), 'wb') as f:
#     pickle.dump([mean_error_per_tasks_n, std_error_per_tasks_n, n_tasks_vec], f)
