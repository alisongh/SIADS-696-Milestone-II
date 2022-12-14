import matplotlib.cm as cm
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score, davies_bouldin_score, calinski_harabasz_score
from time import time
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.cluster.hierarchy import dendrogram


def draw_silhouette(X, k=[2,3], ax='mort_rate', ay='physical_zip_code_codes', random_state=42, sample=20000):

    """
    returns a series of silhouette score & silhouette plot with scatter plot layered with cluster centriods for silhouette analysis

    input:
    X - dataframe to populate silhouette score
    k - range of clusters to plot
    ax - feature on the x axis of the scatter plot
    ay - feature on the y axis of the scatter plot
    random_state - assign random seed for kmeans clustering

    ref: source: https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html#sphx-glr-auto-examples-cluster-plot-kmeans-silhouette-analysis-py

    """

    X = X.sample(frac=sample/len(X.index), replace=False, random_state=random_state)

    for n_clusters in k:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(init='k-means++', n_clusters=n_clusters, random_state=random_state)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels, metric="euclidean")
        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
        )

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(
                np.arange(y_lower, y_upper),
                0,
                ith_cluster_silhouette_values,
                facecolor=color,
                edgecolor=color,
                alpha=0.7,
            )

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(
            X[ax], X[ay], marker=".", s=30, lw=0, alpha=0.7, c=colors, edgecolor="k"
        )

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(
            centers[:, 0],
            centers[:, 1],
            marker="o",
            c="white",
            alpha=1,
            s=200,
            edgecolor="k",
        )

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker="$%d$" % i, alpha=1, s=50, edgecolor="k")

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for "+ax)
        ax2.set_ylabel("Feature space for "+ay)

        plt.suptitle(
            "Silhouette analysis for KMeans clustering on sample data with n_clusters = %d"
            % n_clusters,
            fontsize=14,
            fontweight="bold",
        )

    plt.show()


def draw_cluster_scatter(X, k, random_state=42):

    """
    returns a html page of scatter plot of df X and kmeans cluster as color
    with filter to select which feature to represent x and y

    input:
    X - dataframe
    k - # of clusters
    random_state - assign random seed for kmeans clustering    
    """

    clusterer = KMeans(init='k-means++', n_clusters=k, random_state=random_state)
    cluster_labels = clusterer.fit_predict(X)
    df_seg_kmeans = pd.concat([X.reset_index(drop=True), pd.DataFrame(cluster_labels, columns=['cluster_label'])] , axis =1)
    cols_dd = df_seg_kmeans.columns.tolist()[:-1]

    fig = go.Figure(
        go.Scatter(
            x=df_seg_kmeans[np.random.choice(cols_dd, 1)[0]],
            y=df_seg_kmeans[np.random.choice(cols_dd, 1)[0]],
            mode='markers',
            marker_color=df_seg_kmeans['cluster_label'],
            showlegend=True,
            hovertemplate='x: %{x} <br>y: %{y}',
        )
    )

    fig.update_layout(
        updatemenus=[
            {
                "buttons": [
                    {
                        "label": f"x - {x}",
                        "method": "update",
                        "args": [
                            {"x": [df_seg_kmeans[x]]},
                            {"xaxis": {"title": x}},
                        ],
                    }
                    for x in cols_dd
                ]
            },
            {
                "buttons": [
                    {
                        "label": f"y - {x}",
                        "method": "update",
                        "args": [{"y": [df_seg_kmeans[x]]}, {"yaxis": {"title": x}}],
                    }
                    for x in cols_dd
                ],
                "y": 0.9,
            },
        ],
        margin={"l": 0, "r": 0, "t": 25, "b": 0},
    )
    fig.write_html("kmeans_cluster_scatter.html")


def plot_silhouette(X, kmeans=KMeans(init="k-means++", random_state=42), k_start=2, k_end=20, random_state=42):
    x = [i for i in range(k_start, k_end+1)]
    fit_time = []
    inertia = []
    silhouette = []
    DB = [] #davies-bouldin index
    CH = [] #calinski-harabasz index

    for i in range(k_start, k_end+1):
        name = str(i)+' clusters'
        t0 = time()
        kmeans = KMeans(n_clusters=i, random_state=random_state)
        #estimator = make_pipeline(kmeans).fit(X)
        estimator = kmeans.fit(X)
        fit_time.append(time() - t0)
        inertia.append(estimator.inertia_)

        this_silhouette = silhouette_score(
                X,
                estimator.labels_,
                metric="euclidean",
                sample_size=20000,
                random_state=random_state
            )
        silhouette.append(this_silhouette)

        this_DB = davies_bouldin_score(X, estimator.labels_)
        DB.append(this_DB)

        this_CH = calinski_harabasz_score(X, estimator.labels_)
        CH.append(this_CH)


    fig, (ax_top, ax_mid, ax_btm) = plt.subplots(3, 1, figsize = (7, 8))

    # ax_top

    ax2 = ax_top.twinx()
    ax_top.plot(x, silhouette, color = 'g')
    ax2.plot(x, inertia, color = 'b')

    # giving labels to the axises
    ax_top.set_xlabel('Clusters')
    ax_top.set_ylabel('Silhouette Score', color = 'g')
    ax2.set_ylabel('Inertia', color = 'b')
    ax_top.xaxis.set_ticks(range(k_start,k_end+1))

    # ax_mid

    ax3 = ax_mid.twinx()
    ax_mid.plot(x, DB, color = 'c')
    ax3.plot(x, CH, color = 'm')

    # giving labels to the axises
    ax_mid.set_xlabel('Clusters')
    ax_mid.set_ylabel('Davies-Bouldin Index', color = 'c')
    ax3.set_ylabel('Calinski-Harabasz Index', color = 'm')
    ax_mid.xaxis.set_ticks(range(k_start,k_end+1))

    # ax_btm

    ax_btm.plot(x, fit_time, linestyle = 'dotted', color = 'r')
    ax_btm.set_ylabel('Fitting Time', color = 'r')
    ax_btm.set_xlabel('Clusters')
    ax_btm.xaxis.set_ticks(range(k_start,k_end+1))
    
    fig.suptitle("Cluster Evaluation and Fitting Time for "+str(k_start)+" to "+str(k_end)+" Clusters", y=1)
    fig.tight_layout(h_pad=2, w_pad=2)

    fig.show()


def make_PCA_screeplot(pca):

    """
    input pca fitted object 
    
    returns PCA Scree Plot
    
    """

    colors = ["green" if i >= 1 else "blue" for i in pca.explained_variance_]

    explained = pca.explained_variance_ratio_

    plt.figure(figsize = (10,7))
    plt.plot(range(1, len(explained)+1), explained.cumsum(), marker='o', linestyle = '--')
    plt.bar(range(1, len(explained)+1), explained, color = colors),
    plt.text(15, 0.6, 'green bars: eignevalue >= 1\n blue bars: eigenvalue < 1', fontsize = 14)
    plt.title('Explained Variance by Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.show()

def make_loadings_df(pca_n, feature_names):

    """
    input pcs model and feature names as np array

    returns dataframe of feature loadings by PC
    
    """

    pca_n_loadings = pca_n.components_
    names_pc = ['PC'+str(i) for i in range(1, pca_n.n_components_+1)]

    loadings_df = pd.DataFrame.from_dict(dict(zip(names_pc, pca_n_loadings)))
    loadings_df['variable'] = feature_names
    loadings_df = loadings_df.set_index('variable').round(4).sort_values(by=names_pc, ascending=False)

    return loadings_df

def top_PC_loadings(pca_n, feature_names, top_n=10):

    """
        #top features and loadings for PC

        input pca instance and # of top features to show
    
    """

    loadings_df = make_loadings_df(pca_n, feature_names)

    pc_df = pd.DataFrame()

    for i, col in enumerate(loadings_df.columns):

        top10 = loadings_df[col].sort_values(ascending=False)[:top_n]
        pc_df = pd.concat([pc_df, top10.reset_index().rename(
            columns={
                'variable':col+' Top Feature', 
                col: col+' Loadings (ratio='+str(round(pca_n.explained_variance_ratio_[i],3))+')'
                })], axis=1)

    fig = go.Figure(data=[go.Table(
        columnwidth=[25, 10]*int(len(pc_df.columns)/2),
        header=dict(values=list(pc_df.columns),
                    fill_color='paleturquoise',
                    align='center',
                    font = dict(size=9)
                    ),
        cells=dict(values=pc_df.transpose().values.tolist(),
                fill_color='lavender',
                align=['left', 'center']*int(len(pc_df.columns)/2),
                font = dict(size=9)
                ))
    ])

    fig.show()
    return pc_df


def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)