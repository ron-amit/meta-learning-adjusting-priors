
from __future__ import absolute_import, division, print_function

import os
import torch
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from Models.stochastic_models import get_model
from Utils.common import load_model_state, load_run_data, get_value


matplotlib.rcParams.update({'font.size': 12})
matplotlib.rcParams.update({'lines.linewidth': 2})
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


# -------------------------------------------------------------------------------------------
# Auxilary functions:
#----------------------------------------------------------------------------------------------

def extract_param_list(model, name1, name2):
    # extract parameters which names contain the strings name1 and name2:
    params_per_layer = [named_param for named_param in model.named_parameters()
                        if name1 in named_param[0] and name2 in named_param[0]]
    # note: each element is a tuple (name, values)
    # flatten values to vectors:
    params_per_layer = [(params[0], params[1].view(-1)) for params in params_per_layer]
    return params_per_layer


def log_var_to_sigma(log_var_params):
    return [(named_param[0].replace('_log_var', '_sigma'),
              0.5 * torch.exp(named_param[1]))
             for named_param in log_var_params]

def get_params_statistics(param_list):
    n_list = len(param_list)
    mean_list = np.zeros(n_list)
    std_list = np.zeros(n_list)
    for i_param, named_param in enumerate(param_list):
        param_name = named_param[0]
        param_vals = named_param[1]
        param_mean = get_value(param_vals.mean())
        param_std = get_value(param_vals.std())
        mean_list[i_param] = param_mean
        std_list[i_param] = param_std
        print('Parameter name: {}, mean value: {:.3}, STD: {:.3}'.format(param_name, param_mean, param_std))
    return mean_list, std_list


def plot_statistics(mean_list, std_list, name):

    plt.figure()
    n_list = len(mean_list)
    plt.errorbar(range(n_list), mean_list, yerr=std_list)
    # plt.title("Statistics of the prior {} ".format(name))
    plt.xticks(np.arange(n_list),  fontsize=12)
    plt.xlabel('Layer',  fontsize=16)
    plt.ylabel(name, fontsize=16)


# -------------------------------------------------------------------------------------------
# Analysis function:
#----------------------------------------------------------------------------------------------

def run_prior_analysis(prior_model, showPlt=True):

    # w_mu_params = extract_param_list(prior_model,'_mean', '.w_')
    # b_mu_params = extract_param_list(prior_model,'_mean', '.b_')
    w_log_var_params = extract_param_list(prior_model,'_log_var', '.w_')
    b_log_var_params = extract_param_list(prior_model,'_log_var', '.b_')

    n_layers = len(w_log_var_params)

    # w_sigma_params = log_var_to_sigma(w_log_var_params)
    # b_sigma_params = log_var_to_sigma(b_log_var_params)

    # concatenate weight and bias values:
    log_var_params = []
    for i_layer in range(n_layers):
        values = torch.cat((w_log_var_params[i_layer][1], b_log_var_params[i_layer][1]), 0)
        log_var_params.append(('log_var', values))


    mean_list, std_list = get_params_statistics(log_var_params)

    plot_statistics(mean_list, std_list, name=r'$\log (\sigma^2)$')
    layers_inds = np.arange(n_layers)
    if hasattr(prior_model, 'layers_names'):
        layers_names = [str(i) + ' (' + prior_model.layers_names[i] + ')' for i in layers_inds]
    else:
        layers_names = [str(i) for i in layers_inds]

    plt.xticks(layers_inds, layers_names)

    if showPlt:
        plt.show()


# -------------------------------------------------------------------------------------------
# execute only if run as a script
# -------------------------------------------------------------------------------------------
if __name__ == "__main__":


    root_saved_dir = 'saved/'
    #***** Enter here the relative path to results dir (with the learned prior you want to analyze):
    result_name = 'PermutedLabels_5_Tasks_NewBoundMcAllaster_Comp'
    # result_name = 'Shuffled_200_Pixels_10_Tasks_NewBoundSeeger_Comp'
    result_path = root_saved_dir + result_name
    prm, info_dict = load_run_data(result_path)

    # path to the saved learned meta-parameters
    saved_path = os.path.join(prm.result_dir, 'model.pt')

    # Loads  previously training prior.
    # First, create the model:
    prior_model = get_model(prm)
    # Then load the weights:
    load_model_state(prior_model, saved_path)
    print('Pre-trained  prior loaded from ' + saved_path)

    run_prior_analysis(prior_model, showPlt=False)
    plt.savefig(root_saved_dir + 'layer_variance_' + result_name + '.pdf', format='pdf', bbox_inches='tight')
    plt.show()
