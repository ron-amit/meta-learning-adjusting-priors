from subprocess import call
import argparse

n_train_tasks = 10
n_pixels_shuffles = 200

parser = argparse.ArgumentParser()

parser.add_argument('--complexity_type', type=str,
                    help=" The learning objective complexity type",
                    default='NewBoundSeeger')
# 'NoComplexity' /  'Variational_Bayes' / 'PAC_Bayes_Pentina'   NewBoundMcAllaster / NewBoundSeeger'"

args = parser.parse_args()

complexity_type = args.complexity_type


call(['python', 'main_Meta_Bayes.py',
      '--run-name', 'Shuffled_{}_Pixels_{}_Tasks_{}_Comp'.format(n_pixels_shuffles, n_train_tasks, complexity_type),
      '--data-source', 'MNIST',
      '--data-transform', 'Shuffled_Pixels',
      '--n_pixels_shuffles', str(n_pixels_shuffles),
      '--limit_train_samples_in_test_tasks', '2000',
      '--n_train_tasks',  str(n_train_tasks),
      '--mode', 'MetaTrain',
      '--complexity_type',  complexity_type,
      '--model-name', 'FcNet3',
      '--n_meta_train_epochs', '150',
      '--n_meta_test_epochs', '200',
      '--n_test_tasks', '20',
      '--meta_batch_size', '16',
      ])



