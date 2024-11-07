import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, Latex
from torch.nn import functional as F
os.makedirs('results', exist_ok=True)
from tqdm.notebook import tqdm as tqdm
from sklearn.feature_selection import mutual_info_classif

np.random.seed(42)
# from sklearn.preprocessing import OneHotEncoder

from utils import get_clusters, save_figures, get_ordinations
from models_configs import import_models
from train import process_model

# main call
if __name__ == '__main__':
    print("Running GP-CV")
    # argparse
    import argparse
    parser = argparse.ArgumentParser(description='Run GP-CV')
    parser.add_argument('--n_features', type=int, default=-1, help='Number of features')
    parser.add_argument('--experiment_name', type=str, default='acne_500k', help='Experiment name')
    parser.add_argument('--use_mi', type=int, default=0, help='Use mutual information')
    parser.add_argument('--models_done', type=str, default='', help='')
    parser.add_argument('--n_calls', type=int, default=20, help='Number of calls')
    parser.add_argument('--n_splits', type=int, default=5, help='Number of splits')
    parser.add_argument('--log_neptune', type=int, default=0, help='Log to neptune')
    parser.add_argument('--log_shap', type=int, default=0, help='Log shap values')
    parser.add_argument('--nk_input_features', type=int, default=0, help='Number of input features')

    args = parser.parse_args()
    
    models_done = args.models_done.split(',')

    df = pd.read_csv(f"data/{args.experiment_name}_{args.nk_input_features}k.tsv", sep="\t").transpose()
    experiment_name = f'{args.experiment_name}_{args.n_features}features_mi{args.use_mi}'
    os.makedirs(f"results/{experiment_name}", exist_ok=True)
 
    df.columns = df.iloc[0]
    df = df.drop(df.index[0])
    df = df.astype(float)

    # Keep only the 1000 for testing
    df = df.iloc[:, :args.n_features]
    # Import onehotencoder

    Y = np.array(df.index)
    # remove the dots and everything after control or case in Y
    Y = [y.split('.')[0] for y in Y]
    # onehot encoding
    df.index = Y
    # Make controls 0 and cases 1
    Y = pd.Series([0 if 'control' in y else 1 for y in df.index])
    # Make Y onehot
    X = df
    # Counter(Y)
    # make a progress bar
    if args.use_mi == 1:
        mi = mutual_info_classif(X, Y)
    else:
        mi = np.ones(X.shape[1])
    # Make plots of the mutual information
    # make a progress bar
    sns.histplot(mi, bins=50)
    plt.savefig(f"results/{experiment_name}/mi_hist.png")
    plt.close()
    # Sort the features by mutual information
    from operator import itemgetter
    sorted_mi = sorted(enumerate(mi), key=itemgetter(1))
    sorted_mi = [x[0] for x in sorted_mi]
    X_sorted = X.iloc[:, sorted_mi]
    
    save_figures(X_sorted, Y, experiment_name)
    # get the clusters
    clusters = get_clusters(X_sorted)
    get_ordinations(X_sorted, Y, experiment_name)
    # get the models
    models, hparam_names, spaces = import_models(models_done) 

    data = {"X": X_sorted.copy(), "y": Y.copy(), 'group': pd.Series(list(X.index)), "clusters": clusters[3]}  

    table = [['Model', 'valid_MCC']]
    for model, name in zip(models, list(spaces.keys())):
        display(Markdown(f"##{name}"))
        print(f"### Processing {name}")
        # n_splits=5 serait mieux mais coince
        table = process_model(model, data, mi, name, experiment_name,
                              hparam_names[name], spaces[name], args) 
        fig = plt.figure()

