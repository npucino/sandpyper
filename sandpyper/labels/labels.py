import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.ndimage import gaussian_filter
import scipy.signal as sig


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from tqdm import tqdm_notebook as tqdm

# MODULE____labels


def get_sil_location(merged_df, ks=(2,21),
                     feature_set=["band1","band2","band3","slope"],
                    random_state=10,n_jobs=-1):
    """
    Function to obtain average Silhouette scores for a list of number of clusters (k) in all surveys.
    It uses KMeans as a clusteres and parallel processing for improved speed.

    Warning:
        It might take up to 8h for tables of 6,000,000 observations.

    Args:
        merged_df (Pandas dataframe): The clean and merged dataframe containing the features.
        k_rng (tuple): starting and ending number of clusters to run KMeans and compute SA on.
        feature_set (list): List of strings of features in the dataframe to use for clustering.
        random_state (int): Random seed used to make the randomisation deterministic.
        n_jobs (int): Number of threads to use. Default is -1, which uses all the available cores.

    Returns:
        A dataframe containing average Silhouette scores for each survey, based on the provided feature set.
    """

    # Creates the range of k to be used for Silhouette Analysis
    k_rng=range(*ks)

    # Get list of locations
    list_locs=merged_df.location.unique()

    #Setting up the estimator to scale and translate each feature individually to a 0 to 1 range.
    scaler = MinMaxScaler()

    location_series=[]
    dates_series=[]
    n_clusters_series=[]
    silhouette_avg_series=[]

    for location in tqdm(list_locs):

        list_dates= merged_df.query(f"location=='{location}'").survey_date.unique()

        for survey_date in tqdm(list_dates):
            print(f"Working on : {location}, {survey_date}.")

            data_in=merged_df.query(f"location=='{location}' & survey_date== '{survey_date}'")
            data_in=data_in[feature_set]
            data_in.dropna(inplace=True)


            for n_clusters in tqdm(k_rng):

                minmax_scaled_df = scaler.fit_transform(data_in)
                minmax_scaled_df = np.nan_to_num(minmax_scaled_df)

                clusterer=KMeans(n_clusters=n_clusters, init='k-means++',
                         algorithm='elkan', tol=0.0001,
                         random_state=random_state, n_jobs=-1)

                cluster_labels=clusterer.fit_predict(np.nan_to_num(minmax_scaled_df))

                # The silhouette_score gives the average value for all the samples.
                # This gives a perspective into the density and separation of the formed
                # clusters
                silhouette_avg = silhouette_score(minmax_scaled_df, cluster_labels)
                print("For n_clusters =", n_clusters,
                      "The average silhouette_score is :", silhouette_avg)

                location_series.append(location)
                dates_series.append(survey_date)
                n_clusters_series.append(n_clusters)
                silhouette_avg_series.append(silhouette_avg)


    items_dict = {'location' : pd.Series(data=location_series),
                  'survey_date': pd.Series(data=dates_series),
             'k' : pd.Series(data=n_clusters_series),
                'silhouette_mean': pd.Series(data=silhouette_avg_series)}

    sil_df=pd.DataFrame(items_dict)

    return sil_df


def plot_sil(array, k_rng, feat1=0, feat2=1, random_state=10):
    """
    Function to perform Silhouette Analysis and visualise Silhouette plots iteratively,
    from a list of k (number of clusters).
    [Source:https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html]

    Args:
        array (Numpy array): .
        k_rng (list): List of integers representing the number of clusters k to partition the dataset.
        feat1 (int): The index of the feature to plot on the x-axis.
        feat2 (int): The index of the feature to plot on the y-axis.

    Returns:
        Plots and prints the mean Silhouette score per number of clusters.
    """

    for n_clusters in k_rng:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # used to insert blank space between silohette plots of separate clusters
        ax1.set_ylim([0, len(array) + (n_clusters + 1) * 10])

        clusterer=KMeans(n_clusters=n_clusters, init='k-means++',
                         algorithm='elkan', tol=0.0001,
                         random_state=random_state,n_jobs=-1)
        cluster_labels=clusterer.fit_predict(array)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(array, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(array, cluster_labels)

        y_lower = 10

        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()
            size_cluster_i = ith_cluster_silhouette_values.shape[0]

            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

         # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(array[:, feat1], array[:, feat2], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')

    plt.show()


def get_opt_k (sil_df, sigma=1):
    """
    Function to create a dictionary with optimal number of clusters (k) for all surveys.
    It search for the inflexion points where an additional cluster do not degrade the overall clustering performance.
    It uses a Gaussian smoothed regression of number of k against mean silhouette scores to identify relative minima (first order)
    as possible inlfexion values.
    When multiple relative minimas are found, the smaller k will be the optimal one.
    When no relative minima are found, it searches for peaks in the second order derivative of such regression line.
    If multiple peaks are found, the mean k will be used as optimal.


    Args:
        sil_group_df (Pandas dataframe): Dataframe containing the mean silhouette score per k in each survey.
        sigma (int): Number of standard deviations to use in the Gaussian filter. Default is 1.
    Returns:
        Dictionary with optimal k for each survey.
    """

    si_arr=sil_df.groupby(['location',"survey_date"])["silhouette_mean"].apply(np.array)
    k_arr=sil_df.groupby(['location',"survey_date"])["k"].apply(np.array)

    dict_data={"k_numbers":k_arr,"silhouette_mean":si_arr}
    sil_group_df=pd.DataFrame(dict_data)
    sil_group_df=sil_group_df.reset_index()


    opt_k=dict()


    for i in range(0,sil_group_df.shape[0]):

        location=sil_group_df.iloc[i].location
        survey_date=sil_group_df.iloc[i].survey_date
        sub=sil_group_df.loc[i,["k_numbers","silhouette_mean"]]

        # Passing a gaussian filter to smooth the curve for 1 std sigma
        gauss=gaussian_filter(sub.silhouette_mean, sigma=sigma)

        # obtaining relative minima for the smoothed line (gauss)
        mina=sig.argrelmin(gauss, axis=0, order=1, mode='clip')

        if len(mina[0]) == 0:
        # if no relative minima are found, compute the second order accurate central differences in the interior points
        # as optimal k

            der=np.gradient(gauss)
            peak=sig.find_peaks(der)

            if len(peak)>1:  # if multiple plateu values are found, obtain the mean k of those values

                peak=int(np.mean(peak[0])) +2
                # +2: the peaks of mina values are 0-based index. As k started at 2, adding 2 returns k instead of index
                opt_k[f"{location}_{survey_date}"]=peak

            else:
                opt_k[f"{location}_{survey_date}"]=peak[0][0]+2

        elif len(mina[0]) == 1:

            opt_k[f"{location}_{survey_date}"]=mina[0][0]+2

        else:
            # if multiple relative minimas are found, use the first one as optimal k
            opt_k[f"{location}_{survey_date}"]=mina[0][0] +2

    return opt_k


def kmeans_sa(merged_df,opt_k_dict,thresh_k=5, feature_set=['band1','band2','band3','slope'],
              random_state=10,n_jobs=-1):
    """
    Function to use KMeans on all surveys with the optimal k obtained from the Silhouette Analysis.
    It uses KMeans as a clusterer with parallel processing for improved speed.

    Args:
        merged_df (Pandas dataframe): The clean and merged dataframe containing the features. Must contain the columns point_id, location and survey_date, as well as the
        opt_k_dict (dict): Dictionary containing the optimal k for each survey. See get_opt_k function.
        thresh_k (int): Minimim k to be used. If optimal k is below, then k equals the average k of all above threshold values.
        random_state (int): Random seed used to make the randomisation deterministic.
        n_jobs (int): Number of threads to use. Default is -1, which uses all the available cores.

    Returns:
        A dataframe containing the label_k column, with point_id, location, survey_date and the features used to cluster the data.
    """

    data_merged=merged_df[["point_id","location","survey_date","z","slope","curve","distance","band1","band2","band3"]]
    data_merged.dropna(inplace=True)
    list_locs=data_merged.location.unique()

    scaler = MinMaxScaler()
    data_classified=pd.DataFrame()

    # Set a threshold k, in case a k is lower than 5, use the mean optimal k of the other surveys above threshold
    threshold=5

    # # Compute the mean optimal k of above threshold ks
    arr_k=np.array([i for i in opt_k_dict.values() if i > threshold])
    threshold_k=np.int(np.round(np.mean(arr_k),0))

    for location in tqdm(list_locs):

        list_dates= data_merged.query(f"location=='{location}'").survey_date.unique()

        for survey_date in tqdm(list_dates):

            data_in=data_merged.query(f"location=='{location}'& survey_date=='{survey_date}'")
            data_clean=data_in[feature_set].apply(pd.to_numeric)

            k=opt_k_dict[f"{location}_{survey_date}"]

            if k > threshold:

                minmax_scaled_df = scaler.fit_transform(np.nan_to_num(data_clean))

                clusterer=KMeans(n_clusters=k, init='k-means++',
                         algorithm='elkan', tol=0.0001,
                         random_state=random_state,n_jobs=n_jobs)

                data_in["label_k"]=clusterer.fit_predict(minmax_scaled_df)

                data_classified=pd.concat([data_in,data_classified],ignore_index=True)

            else:

                minmax_scaled_df = scaler.fit_transform(np.nan_to_num(data_clean))

                clusterer=clusterer=KMeans(n_clusters=k, init='k-means++',
                         algorithm='elkan', tol=0.0001,
                         random_state=random_state,n_jobs=n_jobs)

                data_in["label_k"]=clusterer.fit_predict(minmax_scaled_df)

                data_classified=pd.concat([data_in,data_classified],ignore_index=True)

    return data_classified
