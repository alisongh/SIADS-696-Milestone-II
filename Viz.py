import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

def draw_silhouette(X, k=[8, 9, 10, 11, 12, 13, 14], ax='mort_rate', ay='physical_zip_code_codes', random_state=42, sample=20000):

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