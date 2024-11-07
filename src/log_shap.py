import os
import torch
import shap
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import xgboost
# import StratifiedShuffleSplit\
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def interactions_mean_matrix(shap_interactions, X, run, group):
    # Get absolute mean of matrices
    mean_shap = np.abs(shap_interactions).mean(0)
    df = pd.DataFrame(mean_shap, index=X.columns, columns=X.columns)

    # times off diagonal by 2
    df.where(df.values == np.diagonal(df), df.values * 2, inplace=True)

    # display
    plt.figure(figsize=(10, 10), facecolor='w', edgecolor='k')
    sns.set(font_scale=1.5)
    sns.heatmap(df, cmap='coolwarm', annot=True, fmt='.3g', cbar=False)
    plt.yticks(rotation=0)
    f = plt.gcf()
    run[f'shap/interactions_matrix/{group}_values'].upload(f)
    plt.close(f)


def make_summary_plot(df, values, group, run, 
                      log_path, category='explainer',
                      mlops='neptune'):
    shap.summary_plot(values, df, show=False)
    f = plt.gcf()
    if mlops == 'neptune':
        run[f'shap/summary_{category}/{group}_values'].upload(f)

    plt.close(f)


def make_force_plot(df, values, features, group, run, log_path, category='explainer', mlops='neptune'):
    shap.force_plot(df, values, features=features, show=False)
    f = plt.gcf()
    if mlops == 'neptune':
        run[f'shap/force_{category}/{group}_values'].upload(f)

    plt.close(f)


def make_deep_beeswarm(df, values, group, run, log_path, category='explainer', mlops='neptune'):
    shap.summary_plot(values, feature_names=df.columns, features=df, show=False)
    f = plt.gcf()
    if mlops == 'neptune':
        run[f'shap/beeswarm_{category}/{group}_values'].upload(f)

    plt.close(f)


def make_decision_plot(df, values, misclassified, feature_names, group, run, log_path, category='explainer', mlops='neptune'):
    # replace first column with base value
    values = np.c_[values.base_values, values.values]
    shap.decision_plot(df, values, feature_names=list(feature_names),
                       show=False, link='logit')
    f = plt.gcf()
    if mlops == 'neptune':
        run[f'shap/decision_{category}/{group}_values'].upload(f)
        run[f'shap/decision_{category}/{group}_values'].upload(f)
    plt.close(f)


def make_decision_deep(df, values, misclassified, feature_names, group, run, log_path, category='explainer', mlops='neptune'):
    shap.decision_plot(df, values, feature_names=list(feature_names), show=False, link='logit', highlight=misclassified)
    f = plt.gcf()
    if mlops == 'neptune':
        run[f'shap/decision_{category}/{group}_values'].upload(f)
        run[f'shap/decision_{category}/{group}_values'].upload(f)
    plt.close(f)


def make_multioutput_decision_plot(df, values, group, run, log_path, category='explainer', mlops='neptune'):
    shap.multioutput_decision_plot(values, df, show=False)
    f = plt.gcf()
    if mlops == 'neptune':
        run[f'shap/multioutput_decision_{category}/{group}_values'].upload(f)
    plt.close(f)


def make_group_difference_plot(values, mask, group, run, log_path, category='explainer', mlops='neptune'):
    shap.group_difference_plot(values, mask, show=False)
    f = plt.gcf()
    if mlops == 'neptune':
        # run[f'shap/gdiff_{category}/{group}'].upload(f)
        run[f'shap/gdiff_{category}/{group}'].upload(f)
    plt.close(f)


def make_beeswarm_plot(values, group, run, log_path, category='explainer', mlops='neptune'):
    shap.plots.beeswarm(values, max_display=20, show=False)
    f = plt.gcf()
    if mlops == 'neptune':
        # run[f'shap/beeswarm_{category}/{group}'].upload(f)
        run[f'shap/beeswarm_{category}/{group}'].upload(f)
    plt.close(f)


def make_heatmap(values, group, run, log_path, category='explainer', mlops='neptune'):
    shap.plots.heatmap(values, instance_order=values.values.sum(1).argsort(), max_display=20, show=False)
    f = plt.gcf()
    if mlops == 'neptune':
        # run[f'shap/heatmap_{category}/{group}'].upload(f)
        run[f'shap/heatmap_{category}/{group}'].upload(f)
    plt.close(f)


def make_heatmap_deep(values, group, run, log_path, category='explainer', mlops='neptune'):

    shap.plots.heatmap(pd.DataFrame(values), instance_order=values.sum(1).argsort(), max_display=20, show=False)
    f = plt.gcf()
    if mlops == 'neptune':
        # run[f'shap/heatmap_{category}/{group}'].upload(f)
        run[f'shap/heatmap_{category}/{group}'].upload(f)
    plt.close(f)


def make_barplot(df, y, values, group, run, log_path, category='explainer', mlops='neptune'):
    clustering = shap.utils.hclust(df, y, metric='correlation')  # cluster_threshold=0.9
    # shap.plots.bar(values, max_display=20, show=False, clustering=clustering)
    shap.plots.bar(values, max_display=20, show=False, clustering=clustering, clustering_cutoff=0.5)
    f = plt.gcf()
    if mlops == 'neptune':
        run[f'shap/bar_{category}/{group}'].upload(f)
    plt.close(f)


def make_bar_plot(df, values, group, run, log_path, category='explainer', mlops='neptune'):
    shap.bar_plot(values, max_display=40, feature_names=df.columns, show=False)
    f = plt.gcf()
    if mlops == 'neptune':
        # run[f'shap/barold_{category}/{group}'].upload(f)
        run[f'shap/barold_{category}/{group}'].upload(f)
    plt.close(f)

def make_dependence_plot(df, values, var, group, run, log_path, category='explainer', mlops='neptune'):
    shap.dependence_plot(var, values[1], df, show=False)
    f = plt.gcf()
    if mlops == 'neptune':
        run[f'shap/dependence_{category}/{group}'].upload(f)
    plt.close(f)

def log_explainer(run, group, args_dict):
    model = args_dict['model']
    model_name = args_dict['model_name']
    x_df = args_dict['inputs'][group]
    labels = args_dict['labels'][group]
    # columns = args_dict['columns']
    log_path = args_dict['log_path']

    unique_classes = np.unique(labels)
    # The explainer doesn't like tensors, hence the f function
    f = lambda x: model.predict(x)
    X = x_df.to_numpy(dtype=np.float32)

    if model_name == 'xgb':
        model = model.get_booster()
    if model_name in ['xgboost', 'xgb', 'lightgbm', 'rfr', 'rfc']:
        explainer = shap.TreeExplainer(model)
    elif model_name in ['linreg', 'logreg', 'qda', 'lda']:
        # explainer = shap.LinearExplainer(model, x_df, max_evals=2 * x_df.shape[1] + 1)
        explainer = shap.LinearExplainer(model, X)
        # model.set_param({"device": "cuda:0"})
        # dtrain = xgboost.DMatrix(x_df, label=labels)
        # shap_values = model.predict(dtrain, pred_contribs=True)
        # shap_interaction_values = model.predict(dtrain, pred_interactions=True)
    else:
        stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=10, random_state=42)
        indices = stratified_split.split(x_df, labels).__next__()[1]
        x_df = x_df.iloc[indices]
        explainer = shap.KernelExplainer(f, x_df)

    # y_shap = y[test_index]    # Get the shap values from my test data
    shap_values = explainer(X)
    if len(unique_classes) == 2:
        if shap_values.base_values.shape[-1] == 2:
            shap_values_df = pd.DataFrame(
                np.c_[shap_values.base_values[:, 0], shap_values.values[:, :, 0]], 
                columns=['bv'] + list(x_df.columns)
            )
        else:
            shap_values_df = pd.DataFrame(
                np.c_[shap_values.base_values, shap_values.values], 
                columns=['bv'] + list(x_df.columns)
            )
        # Remove shap values that are 0
        shap_values_df = shap_values_df.loc[:, (shap_values_df != 0).any(axis=0)]
        # Save the shap values
        shap_values_df.to_csv(f"{log_path}/{group}_shap.csv")
        shap_values_df = shap_values_df.abs()
        shap_values_df = shap_values_df.sum(0)
        total = shap_values_df.sum()
        shap_values_df = shap_values_df / total
        try:
            # Getting the base value
            bv = shap_values_df['bv']
            label = unique_classes[0]
            # Dropping the base value
            shap_values_df = shap_values_df.drop('bv')
            shap_values_df.to_csv(f"{log_path}/{group}_linear_shap_{label}_abs.csv")
            run[f'shap/linear_{group}_{label}'].upload(f"{log_path}/{group}_linear_shap_{label}_abs.csv")
            shap_values_df.transpose().hist(bins=100, figsize=(10, 10))
            plt.xlabel('SHAP value')
            plt.ylabel('Frequency')
            plt.savefig(f"{log_path}/{group}_linear_shap_{label}_hist_abs.png")
            plt.close()
            plt.title(f'base_value: {np.round(bv, 2)}')
            run[f'shap/linear_{group}_{label}_hist'].upload(f"{log_path}/{group}_linear_shap_{label}_hist_abs.png")
            # start x axis at 0
            shap_values_df.abs().sort_values(ascending=False).plot(kind='kde', figsize=(10, 10))
            # shap_values_df.transpose().cumsum().hist(bins=100, figsize=(10, 10))
            plt.xlim(0, shap_values_df.abs().max())
            plt.xlabel('Density')
            plt.ylabel('Frequency')
            plt.title(f'base_value: {np.round(bv, 2)}')
            plt.savefig(f"{log_path}/{group}_linear_shap_{label}_kde_abs.png")
            plt.close()
            run[f'shap/linear_{group}_{label}_kde'].upload(f"{log_path}/{group}_linear_shap_{label}_kde_abs.png")
            
            values, base = np.histogram(shap_values_df.abs(), bins=40)
            #evaluate the cumulative
            cumulative = np.cumsum(values)
            # plot the cumulative function
            plt.figure(figsize=(12, 8))
            plt.plot(base[:-1], cumulative, c='blue')
            #plot the survival function
            plt.plot(base[:-1], len(shap_values_df.abs())-cumulative, c='green')
            plt.xlabel('SHAP value')
            plt.ylabel('Cumulative Density')
            plt.title(f'base_value: {np.round(bv, 2)}')
            plt.savefig(f"{log_path}/{group}_linear_shap_{label}_cumulative_abs.png")
            plt.close()
            run[f'shap/linear_{group}_{label}_cumulative_abs'].upload(f"{log_path}/{group}_linear_shap_{label}_cumulative_abs.png")
        except:
            pass
    else:
        # save shap_values
        # TODO Verifier que l'ordre est bon
        for i, label in enumerate(unique_classes):
            label = unique_classes[label]
            shap_values_df = pd.DataFrame(
                np.c_[shap_values.base_values[:, i], shap_values.values[:, :, i]], 
                columns=['bv'] + list(x_df.columns)
            )
            # Remove shap values that are 0
            shap_values_df = shap_values_df.loc[:, (shap_values_df != 0).any(axis=0)]

            # Save the shap values
            shap_values_df.to_csv(f"{log_path}/{group}_shap.csv")
            run[f'shap/{group}_{label}'].upload(f"{log_path}/{group}_shap.csv")

            shap_values_df = shap_values_df.abs()
            shap_values_df = shap_values_df.sum(0)
            total = shap_values_df.sum()
            shap_values_df = shap_values_df / total

            # Save the shap values
            shap_values_df.to_csv(f"{log_path}/{group}_shap_abs.csv")
            run[f'shap/{group}_{label}'].upload(f"{log_path}/{group}_shap_abs.csv")

            try:
                # Getting the base value
                bv = shap_values_df['bv']

                # Dropping the base value
                shap_values_df = shap_values_df.drop('bv')
                shap_values_df.to_csv(f"{log_path}/{group}_linear_shap_{label}_abs.csv")
                run[f'shap/linear_{group}_{label}'].upload(f"{log_path}/{group}_linear_shap_{label}_abs.csv")

                shap_values_df.transpose().hist(bins=100, figsize=(10, 10))
                plt.savefig(f"{log_path}/{group}_linear_shap_{label}_hist_abs.png")
                plt.close()
                plt.title(f'base_value: {np.round(bv, 2)}')
                # if i == 0:
                run[f'shap/linear_{group}_{label}_hist'].upload(f"{log_path}/{group}_linear_shap_{label}_hist_abs.png")
                # start x axis at 0
                shap_values_df.abs().sort_values(ascending=False).plot(kind='kde', figsize=(10, 10))
                # shap_values_df.transpose().cumsum().hist(bins=100, figsize=(10, 10))
                plt.xlim(0, shap_values_df.abs().max())
                plt.savefig(f"{log_path}/{group}_linear_shap_{label}_kde_abs.png")
                plt.close()
                plt.title(f'base_value: {np.round(bv, 2)}')
                # if i == 0:
                run[f'shap/linear_{group}_{label}_kde'].upload(f"{log_path}/{group}_linear_shap_{label}_kde_abs.png")
                
                values, base = np.histogram(shap_values_df.abs(), bins=40)
                #evaluate the cumulative
                cumulative = np.cumsum(values)
                # plot the cumulative function
                plt.plot(base[:-1], cumulative, c='blue')
                #plot the survival function
                plt.plot(base[:-1], len(shap_values_df.abs())-cumulative, c='green')

                plt.savefig(f"{log_path}/{group}_linear_shap_{label}_cumulative_abs.png")
                plt.close()
                plt.title(f'base_value: {np.round(bv, 2)}')
                # if i == 0:
                run[f'shap/linear_{group}_{label}_cumulative_abs'].upload(f"{log_path}/{group}_linear_shap_{label}_cumulative_abs.png")

            except:
                pass

    # if x_df.shape[1] <= 1000:
    #     make_barplot(x_df, labels, shap_values[:, :, 0], 
    #                 group, run, 'LinearExplainer', mlops='neptune')
    #     # Summary plot
    #     make_summary_plot(x_df, shap_values[:, :, 0], group, run, 
    #                     'LinearExplainer', mlops='neptune')
    #     make_beeswarm_plot(shap_values[:, :, 0], group, run,
    #                         'LinearExplainer', mlops='neptune')
    #     make_heatmap(shap_values[:, :, 0], group, run,
    #                 'LinearExplainer', mlops='neptune')
    #     # mask = np.array([np.argwhere(x[0] == 1)[0][0] for x in cats])
    #     # make_group_difference_plot(x_df.sum(1).to_numpy(), mask, group, run, 'LinearExplainer', mlops='neptune')
    #     make_bar_plot(x_df, shap_values[0], group, run,
    #                 'LinearExplainer', mlops='neptune')
    #     make_force_plot(x_df, shap_values[0], x_df.columns, 
    #                     group, run, 'LinearExplainer', mlops='neptune')
    return run

def log_kernel_explainer(model, x_df, misclassified, 
                         labels, group, run, cats, log_path):
    unique_classes = np.unique(labels)

    # Convert my pandas dataframe to numpy
    data = x_df.to_numpy(dtype=np.float32)
    data = shap.kmeans(data, 20).data
    # The explainer doesn't like tensors, hence the f function
    explainer = shap.KernelExplainer(model.predict, data)

    # Get the shap values from my test data
    df = pd.DataFrame(data, columns=x_df.columns)
    shap_values = explainer.shap_values(df)
    # shap_interaction = explainer.shap_interaction_values(X_test)
    shap_values_df = pd.DataFrame(np.concatenate(shap_values), columns=x_df.columns)
    for i, label in enumerate(unique_classes):
        if i == len(shap_values):
            break
        shap_values_df.iloc[i].to_csv(f"{log_path}/{group}_kernel_shap_{label}.csv")
    # shap_values = pd.DataFrame(np.concatenate(s))
    # Summary plot
    make_summary_plot(x_df, shap_values, group, run, 'Kernel')

    make_bar_plot(x_df, shap_values_df.iloc[1], group, run, 'localKernel')

    make_decision_plot(explainer.expected_value[0], shap_values[0], misclassified, x_df.columns, group, run, 'Kernel')

    mask = np.array([np.argwhere(x[0] == 1)[0][0] for x in cats])
    make_group_difference_plot(x_df.sum(1).to_numpy(), mask, group, run, 'Kernel')


def log_shap(run, args_dict):
    # explain all the predictions in the test set
    # explainer = shap.KernelExplainer(svc_linear.predict_proba, X_train[:100])
    os.makedirs(args_dict['log_path'], exist_ok=True)
    for group in ['valid', 'test']:
        if group not in args_dict['inputs']:
            continue
        # X = args_dict['inputs'][group]
        # labels = args_dict['labels'][group]
        # X_test_df = pd.DataFrame(X, columns=list(X.columns))

        # TODO Problem with not enough memory...
        try:
            run = log_explainer(run, group, args_dict)
        except:
            print(f"Problem with logging {group}")
            # pass
        # log_kernel_explainer(ae, X_test_df, misclassified,
        #                         best_lists[group]['labels'], group, run, 
        #                         best_lists[group]['labels'], log_path
        #                         )
        return run

