"""Labels module."""

import numpy as np
import os
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.ndimage import gaussian_filter
import scipy.signal as sig

import pandas as pd
import geopandas as gpd
import glob
import matplotlib.pyplot as plt
import seaborn as sb
from sandpyper.outils import coords_to_points

from tqdm.notebook import tqdm

# MODULE____labels


def get_sil_location(merged_df, ks, feature_set, random_state=10):
    """
    Function to obtain average Silhouette scores for a list of number of clusters (k) in all surveys.
    It uses KMeans as a clusteres and parallel processing for improved speed.

    Warning:
        It might take up to 8h for tables of 6,000,000 observations.

    Args:
        merged_df (Pandas dataframe): The clean and merged dataframe containing the features.
        ks (tuple): starting and ending number of clusters to run KMeans and compute SA on.
        feature_set (list): List of strings of features in the dataframe to use for clustering.
        random_state (int): Random seed used to make the randomisation deterministic.

    Returns:
        A dataframe containing average Silhouette scores for each survey, based on the provided feature set.
    """

    # Creates the range of k to be used for Silhouette Analysis
    k_rng = range(*ks)

    # Get list of locations
    list_locs = merged_df.location.unique()

    # Setting up the estimator to scale and translate each feature
    # individually to a 0 to 1 range.
    scaler = MinMaxScaler()

    location_series = []
    dates_series = []
    n_clusters_series = []
    silhouette_avg_series = []

    for location in tqdm(list_locs):

        list_dates = merged_df.query(f"location=='{location}'").raw_date.unique()

        for survey_date_in in tqdm(list_dates):
            print(f"Working on : {location}, {survey_date_in}.")

            data_in = merged_df.query(
                f"location=='{location}' & raw_date == {survey_date_in}"
            )
            data_in = data_in[feature_set]
            data_in.dropna(inplace=True)

            for n_clusters in tqdm(k_rng):

                minmax_scaled_df = scaler.fit_transform(data_in)
                minmax_scaled_df = np.nan_to_num(minmax_scaled_df)

                clusterer = KMeans(
                    n_clusters=n_clusters,
                    init="k-means++",
                    algorithm="elkan",
                    tol=0.0001,
                    random_state=random_state,
                )

                cluster_labels = clusterer.fit_predict(np.nan_to_num(minmax_scaled_df))

                # The silhouette_score gives the average value for all the samples.
                # This gives a perspective into the density and separation of the formed
                # clusters
                silhouette_avg = silhouette_score(minmax_scaled_df, cluster_labels)
                print(
                    "For n_clusters =",
                    n_clusters,
                    "The average silhouette_score is :",
                    silhouette_avg,
                )

                location_series.append(location)
                dates_series.append(survey_date_in)
                n_clusters_series.append(n_clusters)
                silhouette_avg_series.append(silhouette_avg)

    items_dict = {
        "location": pd.Series(data=location_series),
        "raw_date": pd.Series(data=dates_series),
        "k": pd.Series(data=n_clusters_series),
        "silhouette_mean": pd.Series(data=silhouette_avg_series),
    }

    sil_df = pd.DataFrame(items_dict)

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

        clusterer = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            algorithm="elkan",
            tol=0.0001,
            random_state=random_state,
        )
        cluster_labels = clusterer.fit_predict(array)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(array, cluster_labels)
        print(
            "For n_clusters =",
            n_clusters,
            "The average silhouette_score is :",
            silhouette_avg,
        )

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(array, cluster_labels)

        y_lower = 10

        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = sorted(
                sample_silhouette_values[cluster_labels == i]
            )

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

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(
            array[:, feat1],
            array[:, feat2],
            marker=".",
            s=30,
            lw=0,
            alpha=0.7,
            c=colors,
            edgecolor="k",
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
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(
            (
                "Silhouette analysis for KMeans clustering on sample data "
                "with n_clusters = %d" % n_clusters
            ),
            fontsize=14,
            fontweight="bold",
        )

    plt.show()


def get_opt_k(sil_df, sigma=1):
    """
    Function to create a dictionary with optimal number of clusters (k) for all surveys.
    It search for the inflexion points where an additional cluster does not degrade the overall clustering performance.
    It uses a Gaussian smoothed regression of k against mean silhouette scores to identify relative minima (first order)
    as possible inlfexion values.
    When multiple relative minimas are found, the smaller k will be the optimal one.
    When no relative minima are found, it searches for peaks in the second order derivative of such regression line.
    If multiple peaks are found, the mean k, computed so far, will be used as optimal.


    Args:
        sil_df (Pandas dataframe): Dataframe containing the mean silhouette score per k in each survey.
        sigma (int): Number of standard deviations to use in the Gaussian filter. Default is 1.
    Returns:
        Dictionary with optimal k for each survey.
    """

    si_arr = sil_df.groupby(["location", "raw_date"])["silhouette_mean"].apply(
        np.array
    )
    k_arr = sil_df.groupby(["location", "raw_date"])["k"].apply(np.array)

    dict_data = {"k_numbers": k_arr, "silhouette_mean": si_arr}
    sil_group_df = pd.DataFrame(dict_data)
    sil_group_df = sil_group_df.reset_index()

    opt_k = dict()

    for i in range(0, sil_group_df.shape[0]):

        location = sil_group_df.iloc[i].location
        survey_date_in = sil_group_df.iloc[i].raw_date
        sub = sil_group_df.loc[i, ["k_numbers", "silhouette_mean"]]

        # Passing a gaussian filter to smooth the curve for 1 std sigma
        gauss = gaussian_filter(sub.silhouette_mean, sigma=sigma)

        # obtaining relative minima for the smoothed line (gauss)
        mina = sig.argrelmin(gauss, axis=0, order=1, mode="clip")

        if len(mina[0]) == 0:
            # if no relative minima are found, compute the second order accurate central differences in the interior points
            # as optimal k

            der = np.gradient(gauss)
            peak = sig.find_peaks(der)

            if (
                len(peak) > 1
            ):  # if multiple plateu values are found, obtain the mean k of those values

                peak = int(np.mean(peak[0])) + 2
                # +2: the peaks of mina values are 0-based index. As k started at 2, adding 2 returns k instead of index
                opt_k[f"{location}_{survey_date_in}"] = peak

            else:
                opt_k[f"{location}_{survey_date_in}"] = peak[0][0] + 2

        elif len(mina[0]) == 1:

            opt_k[f"{location}_{survey_date_in}"] = mina[0][0] + 2

        else:
            # if multiple relative minimas are found, use the first one as optimal k
            opt_k[f"{location}_{survey_date_in}"] = mina[0][0] + 2

    return opt_k


def kmeans_sa(merged_df, ks, feature_set, thresh_k=5, random_state=10):
    """
    Function to use KMeans on all surveys with the optimal k obtained from the Silhouette Analysis.
    It uses KMeans as a clusterer.

    Args:
        merged_df (Pandas dataframe): The clean and merged dataframe containing the features. Must contain the columns point_id, location and survey_date, as well as the
        opt_k_dict (int, dict): number of clusters (k) or dictionary containing the optimal k for each survey. See get_opt_k function.
        feature_set (list): List of names of features in the dataframe to use for clustering.
        thresh_k (int): Minimim k to be used. If survey-specific optimal k is below this value, then k equals the average k of all above threshold values.
        random_state (int): Random seed used to make the randomisation deterministic.

    Returns:
        A dataframe containing the label_k column, with point_id, location, survey_date and the features used to cluster the data.
    """


    merged_df.dropna(inplace=True)
    list_locs = merged_df.location.unique()

    scaler = MinMaxScaler()
    data_classified = pd.DataFrame()

    # Set a threshold k, in case a k is lower than 5, use the mean optimal k
    # of the other surveys above threshold

    # # Compute the mean optimal k of above threshold ks
    if isinstance(ks, dict):
        arr_k = np.array([i for i in ks.values() if i > thresh_k])
        mean_threshold_k = np.int(np.round(np.mean(arr_k), 0))
    else:
        pass

    for location in tqdm(list_locs):

        list_dates = merged_df.query(f"location=='{location}'").raw_date.unique()

        for survey_date_in in tqdm(list_dates):

            data_in = merged_df.query(
                f"location=='{location}'& raw_date == {survey_date_in}"
            )
            data_clean = data_in[feature_set].apply(pd.to_numeric)

            if isinstance(ks, dict):
                k = ks[f"{location}_{survey_date_in}"]
            else:
                k=ks

            if k < thresh_k:
                k = mean_threshold_k
            else:
                pass

            minmax_scaled_df = scaler.fit_transform(np.nan_to_num(data_clean))

            clusterer = KMeans(
                n_clusters=k,
                init="k-means++",
                algorithm="elkan",
                tol=0.0001,
                random_state=random_state,
            )

            data_in["label_k"] = clusterer.fit_predict(minmax_scaled_df)

            data_classified = pd.concat(
                [data_in, data_classified], ignore_index=True
            )

    return data_classified


def check_dicts_duplicated_values(l_dicts):

    dict_check = {}
    dict_dups = {}
    all_dicts=[dicto for dicto in l_dicts.values()]

    for dict_in in all_dicts:
        for key in set().union(*all_dicts):
            if key in dict_in:
                dict_check.setdefault(key, []).extend(dict_in[key])

    for survey, labels in dict_check.items():
        duplicated=[x for x in labels if labels.count(x) > 1]
        if len(duplicated)>=1:
            dict_dups.update({survey:set(set(duplicated))})

    if len(dict_dups)>0:
        raise ValueError(f"Duplicated label_k found in the following dictionaries.\n\n{dict_dups}\n\nPlease revise and assigned those labels_k to only one class dictionary.")



def classify_labelk(labelled_dataset,l_dicts, cluster_field='label_k', fill_class='sand'):

    check_dicts_duplicated_values(l_dicts)

    labelled_dataset["pt_class"]=np.nan

    all_keys = set().union(*(d.keys() for d in [i for i in l_dicts.values()]))
    class_names=l_dicts.keys()

    classed_df=pd.DataFrame()

    for loc in labelled_dataset.location.unique():
        data_in_loc=labelled_dataset.query(f"location=='{loc}'")[["location","raw_date",cluster_field,"pt_class",'point_id']]

        for raw_date in data_in_loc.raw_date.unique():
            loc_date_tag=f"{loc}_{raw_date}"
            data_in=data_in_loc.query(f"raw_date=={raw_date}")

            if loc_date_tag in all_keys:

                for class_in in class_names:

                    if loc_date_tag in l_dicts[class_in].keys():
                        loc_date_class_values=l_dicts[class_in][loc_date_tag]

                        if len(loc_date_class_values)>=1:
                            tmp_dict={label_k:class_in for label_k in loc_date_class_values}
                            data_in['pt_class'].update(data_in[cluster_field].map(tmp_dict))

                        else:
                            pass
                    else:
                        pass
            else:
                print(f"{loc_date_tag} not in the class dictionaries. All their labels assigned to fill_class {fill_class}.")
                data_in["pt_class"].fillna(fill_class, inplace=True)

            classed_df=pd.concat([classed_df,data_in], ignore_index=True)

    merged=labelled_dataset.iloc[:,:-1].merge(right=classed_df[['point_id','pt_class']], on='point_id', how='left')

    merged["pt_class"].fillna(fill_class, inplace=True)
    return merged


def cleanit(to_clean, l_dicts, cluster_field='label_k', fill_class='sand',
            watermasks_path=None, water_label='water',
            shoremasks_path=None, label_corrections_path=None,
            default_crs={'init': 'epsg:32754'}, crs_dict_string=None,
           geometry_field='coordinates'):

    print("Reclassifying dataset with the provided dictionaries." )
    to_clean_classified=classify_labelk(to_clean, l_dicts)

    if watermasks_path==None and shoremasks_path==None and label_corrections_path==None:
        print("No cleaning polygones have been passed. Returning classified dataset.")
        return to_clean_classified

    processes=[]

    #______ LABELS FINETUNING_______________

    if label_corrections_path != None:
        if os.path.isfile(label_corrections_path):
            label_corrections=gpd.read_file(label_corrections_path)
            print(f"Label corrections provided in CRS: {label_corrections.crs}")
            processes.append("polygon finetuning")
            to_update_finetune=pd.DataFrame()


            for loc in label_corrections.location.unique():
                print(f"Fine tuning in {loc}.")

                to_clean_subset_loc=to_clean_classified.query(f" location == '{loc}'")

                for raw_date in tqdm(label_corrections.query(f"location=='{loc}'").raw_date.unique()):

                    subset_finetune_polys=label_corrections.query(f"location=='{loc}' and raw_date== {raw_date}")

                    for i,row in subset_finetune_polys.iterrows(): # loops through all the polygones

                        target_k=int(row['target_label_k'])
                        new_class=row['new_class']

                        if target_k != 999:
                            data_in=to_clean_subset_loc.query(f"raw_date == {raw_date} and label_k== {target_k}")

                        elif target_k == 999:
                            data_in=to_clean_subset_loc.query(f"raw_date == {raw_date}")

                        selection=data_in[data_in.coordinates.intersects(row.geometry)]

                        if selection.shape[0]==0:
                            selection=data_in[data_in.to_crs(crs_dict_string[loc]).coordinates.intersects(row.geometry)]
                        else:
                            pass
                        selection["finetuned_label"]=new_class

                        print(f"Fine-tuning label_k {target_k} to {new_class} in {loc}-{raw_date}, found {selection.shape[0]} pts.")
                        to_update_finetune=pd.concat([selection,to_update_finetune], ignore_index=True)

            classed_df_finetuned=to_clean_classified.merge(right=to_update_finetune.loc[:,['point_id','finetuned_label']], # Left Join
                                         how='left', validate='one_to_one')

            classed_df_finetuned.finetuned_label.fillna(classed_df_finetuned.pt_class, inplace=True) # Fill NaN with previous sand labels
        else:
            raise NameError("Label correction file path is invalid.")


    else:
        pass

    if shoremasks_path == None and watermasks_path == None:
        print(f"{processes} completed.")

        if 'watermasked_label' in classed_df_finetuned.columns and 'finetuned_label' not in classed_df_finetuned.columns:
            classed_df_finetuned['pt_class']=classed_df_finetuned.watermasked_label
            classed_df_finetuned.drop(['watermasked_label'], axis=1, inplace=True)

        elif 'finetuned_label' in classed_df_finetuned.columns and 'watermasked_label' not in classed_df_finetuned.columns:
            classed_df_finetuned['pt_class']=classed_df_finetuned.finetuned_label
            classed_df_finetuned.drop(['finetuned_label'], axis=1, inplace=True)

        elif 'finetuned_label' in classed_df_finetuned.columns and 'watermasked_label' in classed_df_finetuned.columns:
            classed_df_finetuned['pt_class']=classed_df_finetuned.watermasked_label
            classed_df_finetuned.drop(['finetuned_label','watermasked_label'], axis=1, inplace=True)

        else:
            pass

        return classed_df_finetuned
    else:
        pass

    #______ WATERMASKING_______________

    if watermasks_path != None:
        if os.path.isfile(watermasks_path):
            # apply watermasks
            watermask=gpd.read_file(watermasks_path)
            print(f"watermask  provided in CRS: {watermask.crs}")


            print("Applying watermasks cleaning.")
            processes.append("watermasking")

            if "polygon finetuning" in processes:
                dataset_to_clean=classed_df_finetuned
                starting_labels='finetuned_label'
            else:
                dataset_to_clean=to_clean_classified
                starting_labels='pt_class'


            to_update_watermasked=pd.DataFrame()

            for loc in watermask.location.unique():
                print(f"Watermasking in {loc}.")

                for raw_date in tqdm(watermask.query(f"location=='{loc}'").raw_date.unique()):

                    subset_data=dataset_to_clean.query(f"location=='{loc}' and raw_date == {raw_date}")
                    subset_masks=watermask.query(f"location=='{loc}' and raw_date == {raw_date}")

                    selection=subset_data[subset_data.geometry.intersects(subset_masks.geometry)]
                    if selection.shape[0]==0:
                        selection=subset_data[subset_data.geometry.intersects(subset_masks.to_crs(crs_dict_string[loc]).geometry.any())]
                    else:
                        pass

                    print(f"Setting to {water_label} {selection.shape[0]} pts overlapping provided watermasks.")

                    selection["watermasked_label"]=water_label

                    to_update_watermasked=pd.concat([selection,to_update_watermasked], ignore_index=True)

            classed_df_watermasked=dataset_to_clean.merge(right=to_update_watermasked.loc[:,['point_id','watermasked_label']], # Left Join
                                         how='left', validate='one_to_one')
            classed_df_watermasked.watermasked_label.fillna(classed_df_watermasked.loc[:,starting_labels], inplace=True) # Fill NaN with previous sand labels

            if shoremasks_path == None:
                print(f"{processes} completed.")

                if 'watermasked_label' in classed_df_watermasked.columns and 'finetuned_label' not in classed_df_watermasked.columns:
                    classed_df_watermasked['pt_class']=classed_df_watermasked.watermasked_label
                    classed_df_watermasked.drop(['watermasked_label'], axis=1, inplace=True)

                elif 'finetuned_label' in classed_df_watermasked.columns and 'watermasked_label' not in classed_df_watermasked.columns:
                    classed_df_watermasked['pt_class']=classed_df_watermasked.finetuned_label
                    classed_df_watermasked.drop(['finetuned_label'], axis=1, inplace=True)

                elif 'finetuned_label' in classed_df_watermasked.columns and 'watermasked_label' in classed_df_watermasked.columns:
                    classed_df_watermasked['pt_class']=classed_df_watermasked.watermasked_label
                    classed_df_watermasked.drop(['finetuned_label','watermasked_label'], axis=1, inplace=True)

                else:
                    pass

                return classed_df_watermasked
        else:
            raise NameError("watermask file path is invalid.")

    else:
        pass

    #______ SHOREMASKING_______________

    if shoremasks_path != None:
        if os.path.isfile(shoremasks_path):
            # apply shoremasks
            shoremask=gpd.read_file(shoremasks_path)
            print(f"shoremask  provided in CRS: {shoremask.crs}")
            print("Applying shoremasks cleaning.")
            processes.append("shoremasking")


            if "polygon finetuning" in processes and "watermasking" not in processes:
                dataset_to_clean=classed_df_finetuned
            elif "polygon finetuning" not in processes and "watermasking" in processes:
                dataset_to_clean=classed_df_watermasked
            elif "polygon finetuning"  in processes and "watermasking" in processes:
                dataset_to_clean=classed_df_watermasked
            else:
                dataset_to_clean=to_clean_classified

            inshore_cleaned=gpd.GeoDataFrame()
            for loc in shoremask.location.unique():
                print(f"Shoremasking in {loc}.")

                shore=shoremask.query(f"location=='{loc}'")
                loc_selection=dataset_to_clean.query(f"location=='{loc}'")
                in_shore=loc_selection[loc_selection.geometry.intersects(shore.geometry)]
                if in_shore.shape[0]>=1:
                    pass
                else:
                    in_shore=loc_selection[loc_selection.geometry.intersects(shore.to_crs(crs_dict_string[loc]).geometry.any())]

                print(f"Removing {loc_selection.shape[0] - in_shore.shape[0]} pts falling outside provided shore polygones.")
                inshore_cleaned=pd.concat([in_shore,inshore_cleaned], ignore_index=True)

            print(f"{processes} completed.")

            if 'watermasked_label' in inshore_cleaned.columns and 'finetuned_label' not in inshore_cleaned.columns:
                inshore_cleaned['pt_class']=inshore_cleaned.watermasked_label
                inshore_cleaned.drop(['watermasked_label'], axis=1, inplace=True)

            elif 'finetuned_label' in inshore_cleaned.columns and 'watermasked_label' not in inshore_cleaned.columns:
                inshore_cleaned['pt_class']=inshore_cleaned.finetuned_label
                inshore_cleaned.drop(['finetuned_label'], axis=1, inplace=True)

            elif 'finetuned_label' in inshore_cleaned.columns and 'watermasked_label' in inshore_cleaned.columns:
                inshore_cleaned['pt_class']=inshore_cleaned.watermasked_label
                inshore_cleaned.drop(['finetuned_label','watermasked_label'], axis=1, inplace=True)

            else:
                pass

            return inshore_cleaned
        else:
            raise NameError("shoremask file path is invalid.")
