import torch
import numpy as np
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from skopt import gp_minimize
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.model_selection import train_test_split


from concurrent.futures import ThreadPoolExecutor

def augment_data(X_train, y_train, n_aug, p=0, g=0):
    torch.manual_seed(42)
    X_train2 = X_train.copy()
    y_train2 = y_train.copy()
    colnames = X_train.columns
    # y_train2 = np.array([])
    # X_train2 = np.array([])

    if n_aug > 0:
        for _ in range(n_aug):
            y_train2 = np.concatenate([y_train2, y_train])
            tmp = X_train.copy() + g * np.random.normal(0, 1, X_train.shape)
            tmp = tmp.astype(np.float64)

            arr = torch.from_numpy(np.array(tmp).copy())
            tmp = F.dropout(arr, p).detach().cpu().numpy()
            if len(X_train2) > 0:
                X_train2 = np.concatenate([X_train2, tmp], 0)
            else:
                X_train2 = tmp
    X_train2 = pd.DataFrame(X_train2, columns=colnames)
    return X_train2, y_train2

def get_scaler(scaler):
    if scaler == 'robust':
        return RobustScaler
    elif scaler == 'none' or scaler is None or scaler == 'binary':
        return None
    elif scaler == 'standard':
        return StandardScaler
    elif scaler == 'minmax':
        return MinMaxScaler
    else:
        exit('Wrong scaler name')
        
def save_figures(df, labels, experiment_name):
    # Flatten the matrix to a 1D array for distribution plots
    labels = labels.values
    unique_labels = np.unique(labels)
    data = df.values.flatten()    

    # 1. Histogram
    plt.figure(figsize=(6, 4))
    plt.hist(data, bins=30, edgecolor='black', alpha=0.7)
    plt.title("Histogram of Matrix Values")
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.savefig(f"results/{experiment_name}/histogram_allclasses.png")
    plt.close()
    
    # Make superposed histograms for each class in the same graph
    # plt.figure(figsize=(6, 4))
    # for label in unique_labels:
    #     plt.hist(df[labels == label].values.flatten(), bins=30, alpha=0.7, label=f"Class {label}", edgecolor='black')
    # plt.title("Histogram of Matrix Values")
    # plt.xlabel("Value")
    # plt.ylabel("Frequency")
    # plt.legend()
    # plt.savefig(f"results/{experiment_name}/histogram_perclass.png")
    # plt.close()
    

    # 2. KDE Plot
    # plt.figure(figsize=(6, 4))
    # sns.kdeplot(data, bw_adjust=0.5)
    # plt.title("KDE Plot of Matrix Values")
    # plt.xlabel("Value")
    # plt.ylabel("Density")
    # plt.show()

    # 3. Box Plot
    plt.figure(figsize=(6, 4))
    sns.boxplot(data)
    plt.title("Box Plot of Matrix Values")
    plt.xlabel("Value")
    plt.savefig(f"results/{experiment_name}/boxplot_allclasses.png")
    plt.close()
    
    # boxplot per class
    # plt.figure(figsize=(6, 4))
    # sns.boxplot(data=data, y=labels)
    # plt.title("Box Plot of Matrix Values")
    # plt.xlabel("Class")
    # plt.ylabel("Value")
    # plt.savefig(f"results/{experiment_name}/boxplot_perclass.png")
    # plt.close()
    

    # 4. Violin Plot
    plt.figure(figsize=(6, 4))
    sns.violinplot(data)
    plt.title("Violin Plot of Matrix Values")
    plt.xlabel("Value")
    plt.savefig(f"results/{experiment_name}/violinplot.png")
    plt.close()
    
    # violin plot per class
    # plt.figure(figsize=(6, 4))
    # sns.violinplot(data=df, y=labels)
    # plt.title("Violin Plot of Matrix Values")
    # plt.xlabel("Class")
    # plt.ylabel("Value")
    # plt.savefig(f"results/{experiment_name}/violinplot_perclass.png")
    # plt.close()
    

    # 5. Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(df.values, cmap="viridis")
    plt.title("Heatmap of Matrix")
    plt.savefig(f"results/{experiment_name}/heatmap.png")
    plt.close()
    
    # Heatmap per class
    # plt.figure(figsize=(8, 6))
    # sns.heatmap(df.values, cmap="viridis", yticklabels=labels)
    # plt.title("Heatmap of Matrix")
    # plt.savefig(f"results/{experiment_name}/heatmap_perclass.png")
    # plt.close()
    
    # binarize the data
    df1 = df.applymap(lambda x: 1 if x > 0.5 else 0)
    # 5. Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(df1.values, cmap="viridis")
    plt.title("Heatmap of Matrix")
    plt.savefig(f"results/{experiment_name}/heatmap_binary.png")
    plt.close()

    # Make an histogram of the number of zeros per sample
    plt.hist(np.sum(df == 0, axis=1), bins=20)
    plt.xlabel('Number of zeros')
    plt.ylabel('Number of samples')
    plt.title('Histogram of the number of zeros per sample')
    plt.savefig(f"results/{experiment_name}/histogram_zeros_per_sample_allclasses.png")
    plt.close()
    
    # Make an histogram of the number of zeros per sample per class superposed
    # plt.figure(figsize=(6, 4))
    # for label in unique_labels:
    #     plt.hist(np.sum(df[labels == label] == 0, axis=1), bins=20, alpha=0.7, label=f"Class {label}", edgecolor='black')
    # plt.xlabel('Number of zeros')
    # plt.ylabel('Number of samples')
    # plt.title('Histogram of the number of zeros per sample')
    # plt.legend()
    # plt.savefig(f"results/{experiment_name}/histogram_zeros_per_sample_perclass.png")
    
    # Make an histogram of the number of zeros per feature
    plt.hist(np.sum(df == 0, axis=0), bins=20)
    plt.xlabel('Number of zeros')
    plt.ylabel('Number of features')
    plt.title('Histogram of the number of zeros per feature')
    plt.savefig(f"results/{experiment_name}/histogram_zeros_per_feature_allclasses.png")
    plt.close()
    
    # Make an histogram of the number of zeros per feature per class superposed
    # plt.figure(figsize=(6, 4))
    # for label in unique_labels:
    #     plt.hist(np.sum(df[labels == label] == 0, axis=0), bins=20, alpha=0.7, label=f"Class {label}", edgecolor='black')
    # plt.xlabel('Number of zeros')
    # plt.ylabel('Number of features')
    # plt.title('Histogram of the number of zeros per feature')
    # plt.legend()
    # plt.savefig(f"results/{experiment_name}/histogram_zeros_per_feature_perclass.png")
    # plt.close()
    
 
def get_clusters(X):
    # kmeans with 1 to 10 clusters
    from sklearn.cluster import KMeans
    inertia = []
    clusters = {}
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, n_init='auto', random_state=42)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)
        clusters[i] = kmeans.labels_
    plt.plot(range(1, 11), inertia)
    plt.xlabel('Number of clusters')
    plt.ylabel('Inertia')
    plt.title('Elbow method')
    plt.show()
    plt.close()

    return clusters

def get_ordinations(X, Y, exp_name):
    # Ordinations
    # PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=Y)
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.title('PCA')
    plt.savefig(f"results/{exp_name}/pca.png")
    plt.close()

    # UMAP
    from umap import UMAP
    umap = UMAP(n_components=2)
    X_umap = umap.fit_transform(X)
    plt.scatter(X_umap[:, 0], X_umap[:, 1], c=Y)
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.title('UMAP')
    plt.savefig(f"results/{exp_name}/umap.png")
    plt.close()

    # NMDS
    from sklearn.manifold import MDS
    mds = MDS(n_components=2)
    X_mds = mds.fit_transform(X)
    plt.scatter(X_mds[:, 0], X_mds[:, 1], c=Y)
    plt.xlabel('MDS1')
    plt.ylabel('MDS2')
    plt.title('MDS')
    plt.savefig(f"results/{exp_name}/mds.png")
    plt.close()

    # USE LDA after splitting the data
    lda = LinearDiscriminantAnalysis(n_components=1)
    # train test split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    X_lda = lda.fit_transform(X_train, y_train)
    plt.scatter(X_lda, np.zeros(X_lda.shape), c=y_train)
    plt.xlabel('LDA')
    plt.title('LDA')
    plt.savefig(f"results/{exp_name}/lda.png")
    plt.close()

    # Test scores with LDA
    valid_LDA = lda.transform(X_test)
    test_score = lda.score(X_test, y_test)
    # MCC
    test_mcc = metrics.matthews_corrcoef(y_test, lda.predict(X_test))
    train_mcc = metrics.matthews_corrcoef(y_train, lda.predict(X_train))
    print('Test score with LDA:', test_mcc)
    print('Train score with LDA:', train_mcc)
    plt.scatter(valid_LDA, np.zeros(valid_LDA.shape), c=y_test)
    plt.xlabel('LDA')
    plt.title('LDA')
    plt.savefig(f"results/{exp_name}/lda_test.png")
    plt.close()

    # Test scores with QDA
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(X_train, y_train)
    train_score = qda.score(X_train, y_train)
    test_score = qda.score(X_test, y_test)
    # valid_QDA = qda.transform(X_test)
    # MCC
    test_mcc = metrics.matthews_corrcoef(y_test, qda.predict(X_test))
    train_mcc = metrics.matthews_corrcoef(y_train, qda.predict(X_train))
    print('Test score with QDA:', test_mcc)
    print('Train score with QDA:', train_mcc)
    # plt.scatter(valid_QDA, np.zeros(valid_LDA.shape), c=y_test)
