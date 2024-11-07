import copy
import os
import neptune
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from tqdm.notebook import tqdm
from log_shap import log_shap
from utils import get_scaler, augment_data
from skopt.plots import plot_convergence, plot_evaluations, plot_objective, plot_regret
from skopt import gp_minimize

# TODO NEED TO CHANGE THE ACTUAL KEY BY A GLOBAL VARIABLE
NEPTUNE_API_TOKEN = "eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJlZjRiZGUzYS1kNTJmLTRkNGItOWU1MS1iNDU3MGE1NjAyODAifQ=="
NEPTUNE_PROJECT_NAME = "METAG/"
NEPTUNE_MODEL_NAME = 'AC-'

class Train:
    def __init__(self, model, model_name, exp_name, data, hparams_names, mi, args):
        self.model = model
        self.name = model_name
        self.exp_name = exp_name
        self.data = data
        self.hparams_names = hparams_names
        self.h_params = None
        self.best_test_scores = None
        self.n_splits = args.n_splits
        self.mi = mi
        self.use_mi = args.use_mi
        self.iter = 0
        self.pbar = tqdm(range(args.n_calls), position=0, leave=True)
        self.best_score = -np.inf
        self.log_neptune = args.log_neptune
        self.log_shap = args.log_shap
        if 'bagging' in model_name:
            self.bagging = True
        else:
            self.bagging = False
        self.nk_input_features = args.nk_input_features
        self.best_model = None
        self.best_hparams = None
        self.n_calls = args.n_calls
        self.args = args
        
    def init_neptune(self, h_params):
        run = neptune.init_run(
            project=f"{NEPTUNE_PROJECT_NAME}{self.exp_name.split('_')[0].upper()}",
            api_token=NEPTUNE_API_TOKEN,
            source_files=['gp_cv.py',
                            'models_config.py',
                            'utils.py',
                            'train.py',
                            'requirements.txt',
                            'log_shap.py',
                            ],
        )
        h_params_dict = {
            hparam: h_params[i] for i, hparam in enumerate(self.hparams_names)
        }
        run["parameters"] = h_params_dict
        run['name'] = self.name
        run['use_mi'] = self.use_mi
        run['nk_input_features'] = self.nk_input_features
        run['n_splits'] = self.n_splits
        run['n_features'] = self.args.n_features
        return run, h_params_dict


    def train(self, h_params):
        if self.log_neptune:
            # Create a Neptune run object
            run, h_params_dict = self.init_neptune(h_params)
        self.iter += 1
        bag = 1
        features_cutoff = None
        param_grid = {}
        ensemble_grid = {}
        n_aug = 3  # default if not specified in hparams
        scaler = 'standard' # default if not specified in hparams
        for name, param in zip(self.hparams_names, h_params):
            # The number of augmentation to use is optimized
            if self.bagging:
                if name != 'n_aug' and bag:
                    ensemble_grid[name] = param
                    continue
                else:
                    bag = 0
            if name == 'n_aug':
                n_aug = param
            elif name == 'p':
                p = param
            elif name == 'g':
                g = param
            # Many scalers are "optimized" during the hparams optimization. 
            elif name == 'scaler':
                scaler = param
            elif name == 'features_cutoff':
                features_cutoff = param
            elif name == 'zeros_cutoff':
                zeros_cutoff = param
            else:
                param_grid[name] = param        
        # print hparams and their names

        X = self.data['X'].copy()
        clusters = self.data['clusters']
        # Keep features with mi only up to features_cutoff. mis are ordered in descending order
        features_cutoff = int(X.shape[1] * features_cutoff)
        if self.use_mi:
            X = X.iloc[:, self.mi[:features_cutoff]].copy()

        # This function returns the scaler from a string
        scaler = get_scaler(scaler)

        X = X.loc[:, (X != 0).mean() > zeros_cutoff].copy()
        if X.shape[0] == 0:
            run['log_shap'] = 0
            return 1
        if scaler is not None and scaler != 'none' and scaler != 'binary':
            scaler = scaler()
            try:
                X.iloc[:] = scaler.fit_transform(X)
            except:
                run['log_shap'] = 0
                return 1
        elif scaler == 'binary':
            X = X.applymap(lambda x: 1 if x > 0.5 else 0)
        # Shuffle is False by default, but just in case it ever changes
        # Shuffle needs to be False. For each iteration of the hparam optimization, the splits must always 
        # be exactly the same for a fair comparison. 
        scores = { 
            group: {
                'acc': [], 
                'mcc': [],
                'ari': [],
                'ami': [],
                'precision': [],
                'TPR': [],
                'FPR': [],
                'TNR': [],
                'FNR': [],
                'informedness': [],
                'kappa': [],
                'f1': [],
                'cluster_metrics':{f'{cluster_id}_acc': [] for cluster_id in np.unique(clusters)},
            } for group in ['train', 'valid', 'test']
        }
        preds_dict = {group: [] for group in ['train', 'valid', 'test']}
        probas_dict = {group: [] for group in ['train', 'valid', 'test']}
        probas_dict_true = {group: [] for group in ['train', 'valid', 'test']}
        ys_dict = {group: [] for group in ['train', 'valid', 'test']}
        h = 0
        while h < self.n_splits:                                
            print(h)
            skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
            train_nums = np.arange(0, len(self.data['y']))
            splitter = skf.split(train_nums, self.data['y'], self.data['clusters'])
            if h > 0 and h < self.n_splits - 1:
                for i in range(h):
                    _, _ = splitter.__next__()
                _, valid_inds = splitter.__next__()
                _, test_inds = splitter.__next__()

            elif h == self.n_splits - 1:
                _, test_inds = splitter.__next__()
                for i in range(h-1):
                    _, _ = splitter.__next__()
                _, valid_inds = splitter.__next__()
            else:
                _, valid_inds = splitter.__next__()
                _, test_inds = splitter.__next__()

            train_inds = [x for x in train_nums if x not in np.concatenate((valid_inds, test_inds))]

            classifier = self.model
            if self.bagging:
                classifier.base_estimator.set_params(**ensemble_grid)
            classifier.set_params(**param_grid)
            ys = {group: self.data['y'].iloc[inds].copy() for group, inds in \
                  zip(['train', 'valid', 'test'], [train_inds, valid_inds, test_inds])}
            cs = {group: clusters[inds] for group, inds in \
                    zip(['train', 'valid', 'test'], [train_inds, valid_inds, test_inds])}
            Xs = {group: X.iloc[inds].copy() for group, inds in \
                    zip(['train', 'valid', 'test'], [train_inds, valid_inds, test_inds])}
            
            X_train2, y_train2 = augment_data(Xs['train'], ys['train'], n_aug, p, g)
            if X_train2.shape[1] == 0:
                run['log_shap'] = 0
                return 1
            classifier = classifier.fit(X_train2.values, y_train2)
            preds = {group: classifier.predict(Xs[group].values) for group in ['train', 'valid', 'test']}
            try:
                probas_preds = {group: np.max(classifier.predict_proba(Xs[group].values), 1) for group in ['train', 'valid', 'test']}
                probas_true = {group: classifier.predict_proba(Xs[group].values) for group in ['train', 'valid', 'test']}
                for group in ['train', 'valid', 'test']:
                    probas_true[group] = [x[y] for x, y in zip(probas_true[group], ys[group])]
            except:
                probas_preds = preds
                probas_true = preds
            for group in ['train', 'valid', 'test']:
                preds_dict[group] += list(preds[group])
                ys_dict[group] += list(ys[group])
                probas_dict[group] += list(probas_preds[group])
                probas_dict_true[group] += list(probas_true[group])
            for group in ['train', 'valid', 'test']:
                scores[group]['mcc'] += [metrics.matthews_corrcoef(ys[group], preds[group])]
                scores[group]['acc'] += [accuracy_score(ys[group], preds[group])]
                scores[group]['ari'] += [metrics.adjusted_rand_score(ys[group], cs[group])]
                scores[group]['ami'] += [metrics.adjusted_mutual_info_score(ys[group], preds[group])]
                scores[group]['precision'] += [metrics.precision_score(ys[group], preds[group], average='binary', zero_division=0)]
                scores[group]['TPR'] += [metrics.recall_score(ys[group], preds[group], average='binary')]
                scores[group]['f1'] += [metrics.f1_score(ys[group], preds[group], average='binary')]
                scores[group]['TNR'] += [metrics.recall_score(ys[group], preds[group], pos_label=0, average='binary')]
                scores[group]['FNR'] += [1 - metrics.recall_score(ys[group], preds[group], average='binary')]
                scores[group]['FPR'] += [1 - metrics.recall_score(ys[group], preds[group], pos_label=0, average='binary')]
                scores[group]['informedness'] += [scores[group]['TPR'][-1] + scores[group]['TNR'][-1] - 1]
                scores[group]['kappa'] += [metrics.cohen_kappa_score(ys[group], preds[group])]
            
            c_dict = {group: cs[group] for group in ['train', 'valid', 'test']}
            y_dict = {group: ys[group] for group in ['train', 'valid', 'test']}
            x_dict = {group: Xs[group] for group in ['train', 'valid', 'test']}
            all_indices = []
            for cluster_id in np.unique(clusters):
                for group in ['train', 'valid', 'test']:
                    # cluster_mask = (c_dict[group] == cluster_id)
                    cluster_mask = np.argwhere(c_dict[group] == cluster_id).flatten()
                    if len(cluster_mask) == 0:
                        print(f'Empty cluster: {group} {cluster_id}')
                        continue
                    y_cluster = y_dict[group].iloc[cluster_mask].values
                    all_indices += list(cluster_mask)
                    # print(len(cluster_mask))
                    try:
                        if x_dict[group].iloc[cluster_mask].shape[0] == 1:
                            y_pred = classifier.predict(x_dict[group].iloc[cluster_mask].values.reshape(1, -1))
                        else:
                            y_pred = classifier.predict(x_dict[group].iloc[cluster_mask].values)
                    except:
                        y_pred = classifier.predict(x_dict[group].iloc[cluster_mask])
                    try:
                        acc = accuracy_score(y_cluster, y_pred)
                    except:
                        pass
                    try:
                        mcc = metrics.matthews_corrcoef(y_cluster, y_pred)
                    except:
                        pass
                    scores[group]['cluster_metrics'][f'{cluster_id}_acc'] += [acc]

            if self.log_neptune:
                for group in scores.keys():
                    for metric in scores[group].keys():
                        if metric != 'cluster_metrics':
                            run[f'{group}/{metric}'].log(scores[group][metric][-1])
                        else:
                            for cluster_metric in scores[group][metric].keys():
                                try:
                                    run[f'{group}/{cluster_metric}'].log(scores[group][metric][cluster_metric][-1])
                                except:
                                    pass
            
            h += 1

        score = np.mean(scores['valid']['mcc']) 

        self.pbar.update(1)
        if self.log_neptune:
            self.log_stuff(run, h_params_dict, ys_dict, preds_dict, probas_dict_true, scores)
        if score > self.best_score:
            self.best_model = copy.deepcopy(classifier)
            self.best_score = score
            self.best_scores = copy.deepcopy(scores)
            os.makedirs(f'results/{self.exp_name}', exist_ok=True)
            np.save(f'results/{self.exp_name}/best_test_scores_{self.name}', scores['test']['mcc'])
            self.best_hparams = copy.deepcopy(h_params)
        self.pbar.set_description("Best score: %s" % np.round(self.best_score, 3))
        print('Best score:', self.best_score)
        # Print groups mcc
        print(f'Current mccs: train: {np.mean(scores["train"]["mcc"])}, valid: {np.mean(scores["valid"]["mcc"])}, test: {np.mean(scores["test"]["mcc"])}')

        if self.log_neptune:
            run['log_shap'] = 0
            run['iter'] = self.iter
            # End the run
            run.stop()
        if self.iter == self.n_calls:
            if self.log_neptune and self.log_shap:
                run, h_params_dict = self.init_neptune(self.best_hparams)
                run['log_shap'] = 1

                features_cutoff = int(X.shape[1] * h_params_dict['features_cutoff'])
                X = self.data['X'].copy()
                if self.use_mi:
                    X = X.iloc[:, self.mi[:features_cutoff]].copy()

                # This function returns the scaler from a string
                scaler = get_scaler(h_params_dict['scaler'])

                X = X.loc[:, (X != 0).mean() > h_params_dict['zeros_cutoff']].copy()
                if X.shape[0] == 0:
                    run['log_shap'] = 0
                    return 1
                if scaler is not None and scaler != 'none' and scaler != 'binary':
                    scaler = scaler()
                    try:
                        X.iloc[:] = scaler.fit_transform(X)
                    except:
                        run['log_shap'] = 0
                        return 1
                elif scaler == 'binary':
                    X = X.applymap(lambda x: 1 if x > 0.5 else 0)


                Xs = {group: X.iloc[inds].copy() for group, inds in \
                        zip(['train', 'valid', 'test'], [train_inds, valid_inds, test_inds])}
                args_dict = {
                    'inputs': Xs,
                    'labels': ys,
                    'model': self.best_model,
                    'model_name': self.name,
                    'log_path': f'logs/{self.exp_name}',
                }
                self.log_stuff(run, h_params_dict, ys_dict, preds_dict, probas_dict_true, scores)
                run = log_shap(run, args_dict)
                run.stop()

        return -score
    
    def log_stuff(self, run, h_params_dict, ys_dict, preds_dict, probas_dict_true, scores):
        # Change cluster_metrics only to remove a level of depth in the dict. for example, cluster_metrics[0] becomes cluster_metrics_0
        for group in ['train', 'valid', 'test']:
            if 'cluster_metrics' in scores[group].keys():
                for cluster_metric in scores[group]['cluster_metrics'].keys():
                    scores[group][f'cluster_{cluster_metric}'] = scores[group]['cluster_metrics'][cluster_metric]
                scores[group].pop('cluster_metrics')

        # Save what is in scores as a json
        pd.DataFrame(scores).to_json(f'results/{self.exp_name}/scores_{self.name}.json')
        # Save what is in score, except for cluster_metrics, as a table in csv format
        pd.DataFrame({k: v for k, v in scores.items()}).to_csv(f'results/{self.exp_name}/scores_{self.name}.csv')
        # Save scores after getting averages
        scores = {group: {k: np.mean(v) for k, v in scores[group].items()} for group in scores.keys()}
        pd.DataFrame(scores).to_json(f'results/{self.exp_name}/scores_avg_{self.name}.json')
        pd.DataFrame(scores).to_csv(f'results/{self.exp_name}/scores_avg_{self.name}.csv')

        # Save the best hparams to file
        pd.DataFrame(h_params_dict, index=[0]).to_csv(f'results/{self.exp_name}/best_hparams_{self.name}.csv')
        # Save confusion matrices
        for group in ['train', 'valid', 'test']:
            cm = confusion_matrix(ys_dict[group], preds_dict[group])
            pd.DataFrame(cm).to_csv(f'results/{self.exp_name}/confusion_matrix_{group}_{self.name}.csv')
            labels = np.unique(ys_dict[group])

            # Plot de la matrice de confusion
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
            plt.xlabel("Predicted Labels")
            plt.ylabel("True Labels")
            plt.title(f"{group} mcc: {np.round(scores[group]['mcc'], 3)}, acc: {np.round(scores[group]['acc'], 3)}")

            # Sauvegarder le graphique dans un fichier
            plt.savefig(f"results/{self.exp_name}/confusion_matrix_{group}_{self.name}.png")
            plt.close()

            # Save scatter plot of the predictions, small dots.
            # Different colors for different ys
            plt.figure(figsize=(8, 6))
            
            unique_labels = np.unique(ys_dict[group])
            for label in unique_labels:
                mask_true = ys_dict[group] == label
                # Add jitter to x
                x_jitter = np.array(preds_dict[group])[mask_true] + np.random.normal(0, 0.1, np.sum(mask_true))
                plt.scatter(x=x_jitter,
                            y=np.array(probas_dict_true[group])[mask_true],
                            label=f'True {label}', s=1)
            plt.legend()
            plt.xlabel('True labels')
            plt.ylabel('Probability labels')
            plt.title(f'{group} predictions')
            plt.savefig(f"results/{self.exp_name}/scatter_{group}_{self.name}.png")
            plt.close()
        
        run[f"scores/{self.name}"].upload(f"results/{self.exp_name}/scores_{self.name}.csv")
        run[f"scores_avg/{self.name}"].upload(f"results/{self.exp_name}/scores_avg_{self.name}.csv")
        run[f"best_hparams/{self.name}"].upload(f"results/{self.exp_name}/best_hparams_{self.name}.csv")
        
        run[f"confusion_matrix_train/{self.name}"].upload(f"results/{self.exp_name}/confusion_matrix_train_{self.name}.csv")
        run[f"confusion_matrix_valid/{self.name}"].upload(f"results/{self.exp_name}/confusion_matrix_valid_{self.name}.csv")
        run[f"confusion_matrix_test/{self.name}"].upload(f"results/{self.exp_name}/confusion_matrix_test_{self.name}.csv")
        
        run[f"confusion_matrix_train/{self.name}"].upload(f"results/{self.exp_name}/confusion_matrix_train_{self.name}.png")
        run[f"confusion_matrix_valid/{self.name}"].upload(f"results/{self.exp_name}/confusion_matrix_valid_{self.name}.png")
        run[f"confusion_matrix_test/{self.name}"].upload(f"results/{self.exp_name}/confusion_matrix_test_{self.name}.png")

        run[f"scatter_train/{self.name}"].upload(f"results/{self.exp_name}/scatter_train_{self.name}.png")
        run[f"scatter_valid/{self.name}"].upload(f"results/{self.exp_name}/scatter_valid_{self.name}.png")
        run[f"scatter_test/{self.name}"].upload(f"results/{self.exp_name}/scatter_test_{self.name}.png")
        
    def get_shap(self):
        log_shap(self.best_model,
                 self.name,
                 self.data['X'], # TODO change log_shap to accept self.data
                 self.data['y'], # TODO change log_shap to accept self.data
                 self.data['X'].columns, # TODO change log_shap to accept self.data
                 f'logs/{self.exp_name}')

def process_model(model, data, mi, model_name, exp_name, hp, space, args):
    train = Train(model, model_name, exp_name, data, hp, mi, args)
    res = gp_minimize(train.train, space, n_calls=args.n_calls, random_state=41)

    # train.get_shap()

    plot_convergence(res)
    plt.show()
    try:
        plot_evaluations(res)
        plt.show()
    except:
        pass
    try:
        plot_objective(res)
        plt.show()
    except:
        pass
    try:
        plot_regret(res)
        plt.show()
    except:
        pass
    
    # param_grid = {}
    # for name, param in zip(hp, res.x):
    #     if name == 'n_aug':
    #         n_aug = param
    #     elif name == 'scaler':
    #         scaler = param
    #     else:
    #         param_grid[name] = param
    # scaler = get_scaler(scaler)()
    return res
   