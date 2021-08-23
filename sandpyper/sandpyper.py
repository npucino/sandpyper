"""Main Sandpyper module defining the ProfileSet and ProfileDynamics classes with their analytical and plotting methods.
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import pickle
import re
import datetime
import seaborn as sb
import matplotlib.pyplot as plt
from itertools import product as prod

from pysal.explore.giddy.markov import Markov
from pysal.viz.mapclassify import (EqualInterval,
                                   FisherJenks,
                                   HeadTailBreaks,
                                   JenksCaspall,
                                   KClassifiers,
                                   Quantiles,
                                   Percentiles,
                                   UserDefined)

import shapely

from sandpyper.common import (
    compute_multitemporal,
    LISA_site_level,
    kmeans_sa,
    cleanit,
    cross_ref,
    create_spatial_id,
    create_id,
    filter_filename_list,
    getListOfFiles,
    getDate,
    getLoc,
    check_dicts_duplicated_values,
    spatial_id,
    create_details_df,
    get_coastal_Markov,
    compute_multitemporal,
    get_lod_table,
    plot_lod_normality_check,
    get_rbcd_transect,
    get_state_vol_table,
    get_transects_vol_table,
    plot_alongshore_change,
    plot_mec_evolution,
    plot_single_loc,
    getListOfFiles,
    getLoc,
    get_terrain_info,
    get_elevation,
    get_raster_px,
    get_dn,
    get_profiles,
    get_profile_dn,
    extract_from_folder)

class ProfileSet():

    def __init__(self,
                 dirNameDSM,
                 dirNameOrtho,
                 dirNameTrans,
                 transects_spacing,
                 loc_codes,
                 loc_search_dict,
                 crs_dict_string,
                check='all'):
        """This class sets up the monitoring global parameters, input files directories and creates a check dataframe to confirm all CRSs and files are matched up correctly.

        Args:
            dirNameDSM (str): Path of the directory containing the DSM datasets.
            dirNameOrtho (str): Path of the directory containing the orthophotos datasets.
            dirNameTrans (str): Path of the directory containing the transect files (.gpkg, .shp).
            transects_spacing (float): The alonghsore spacing between transects.
            loc_codes (list): List of strings of location codes.
            loc_search_dict (dict): A dictionary where keys are the location codes and values are lists containing the expected full location string, including the location code itself (["wbl","Warrnambool", "warrnambool","warrny"]).
            crs_dict_string (dict): Dictionary storing location codes as key and crs information as values, in dictionary form.
            check (str, optional): If 'all', the check dataframe will contain both DSMs and orthophotos information. If one of 'dsm' or 'ortho', only check the desired data type.

        Returns:
            object: ProfileSet object.
        """


        self.dirNameDSM=dirNameDSM
        self.dirNameOrtho=dirNameOrtho
        self.dirNameTrans=dirNameTrans
        self.transects_spacing=transects_spacing

        self.loc_codes=loc_codes
        self.loc_search_dict=loc_search_dict
        self.crs_dict_string=crs_dict_string

        if check=="dsm":
            path_in=self.dirNameDSM
        elif check == "ortho":
            path_in=self.dirNameOrtho
        elif check == "all":
            path_in=[self.dirNameDSM, self.dirNameOrtho]


        self.check=cross_ref(path_in,
                        self.dirNameTrans,
                        print_info=True,
                        loc_search_dict=self.loc_search_dict,
                        list_loc_codes=self.loc_codes)

    def save(self, name, out_dir):
        """Save object using pickle.

        Args:
            name (str): Name of the file to save.
            out_dir (str): Path to the directory where to save the object.

        Returns:
            file (pkl): pickle file.
        """

        savetxt=f"{os.path.join(out_dir,name)}.p"
        pickle.dump( self, open( savetxt, "wb" ) )
        print(f"ProfileSet object saved in {savetxt} .")

    def extract_profiles(self,
                         mode,
                         tr_ids,
                         sampling_step,
                         lod_mode,
                         add_xy,
                         add_slope=False,
                         default_nan_values=-10000):
        """Extract pixel values from orthophotos, DSMs or both, along transects in all surveys as a GeoDataFrame stored in the ProfileSet.profiles attribute.

        Args:
            mode (str): If 'dsm', extract from DSMs. If 'ortho', extracts from orthophotos. if "all", extract from both.
            tr_ids (str): The name of the field in the transect file that is used to store the transects ID.
            sampling_step (float): Distance along-transect to extract data points from. In meters.
            lod_mode (): If a valid path to a folder storing the LoD transects is provided, extracts elevation data (error) along those transects. If a value is provided, use this value across all surveys. If set to zero, then no LoD will be used.
            add_xy (bool): If True, adds extra columns with long and lat coordinates in the input CRS.
            add_slope (bool): If True, computes slope raster in degrees (increased processing time) and extract slope values across transects.
            default_nan_values (int): Value used for NoData specification in the rasters used. In Pix4D, this is -10000 (default).

         Returns:
            attributes (pd.DataFrame): adds an attribute (.profiles) to the ProfileSet object and if lod_mode is active adds another attribute (.lod) with the elevation change along LoD transects.
        """

        if mode=="dsm":
            path_in=self.dirNameDSM
        elif mode == "ortho":
            path_in=self.dirNameOrtho
        elif mode == "all":
            path_in=[self.dirNameDSM,self.dirNameOrtho]
        else:
            raise NameError("mode must be either 'dsm','ortho' or 'all'.")

        if mode in ["dsm","ortho"]:

            profiles=extract_from_folder(dataset_folder=path_in,
                transect_folder=self.dirNameTrans,
                tr_ids=tr_ids,
                mode=mode,sampling_step=sampling_step,
                list_loc_codes=self.loc_codes,
                add_xy=add_xy,
                add_slope=add_slope,
                default_nan_values=default_nan_values)

            profiles["distance"]=np.round(profiles.loc[:,"distance"].values.astype("float"),2)

        elif mode == "all":

            print("Extracting elevation from DSMs . . .")
            profiles_z=extract_from_folder( dataset_folder=path_in[0],
                    transect_folder=self.dirNameTrans,
                    mode="dsm",
                    tr_ids=tr_ids,
                    sampling_step=sampling_step,
                    list_loc_codes=self.loc_codes,
                    add_xy=add_xy,
                    add_slope=add_slope,
                    default_nan_values=default_nan_values )

            print("Extracting rgb values from orthos . . .")
            profiles_rgb=extract_from_folder(dataset_folder=path_in[1],
                transect_folder=self.dirNameTrans,
                tr_ids=tr_ids,
                mode="ortho",sampling_step=sampling_step,
                list_loc_codes=self.loc_codes,
                add_xy=add_xy,
                default_nan_values=default_nan_values)

            profiles_rgb["distance"]=np.round(profiles_rgb.loc[:,"distance"].values.astype("float"),2)
            profiles_z["distance"]=np.round(profiles_z.loc[:,"distance"].values.astype("float"),2)

            profiles_merged = pd.merge(profiles_z,profiles_rgb[["band1","band2","band3","point_id"]],on="point_id",validate="one_to_one")
            profiles_merged=profiles_merged.replace("", np.NaN)
            profiles_merged['z']=profiles_merged.z.astype("float")

            self.profiles=profiles_merged

        else:
            raise NameError("mode must be either 'dsm','ortho' or 'all'.")

        self.sampling_step=sampling_step


        if os.path.isdir(lod_mode):
            if mode == 'dsm':
                lod_path_data=self.dirNameDSM
            elif mode == 'all':
                lod_path_data=path_in[0]

            print("Extracting LoD values")

            lod=extract_from_folder( dataset_folder=lod_path_data,
                    transect_folder=lod_mode,
                    tr_ids=tr_ids,
                    mode="dsm",
                    sampling_step=sampling_step,
                    list_loc_codes=self.loc_codes,
                    add_xy=False,
                    add_slope=False,
                    default_nan_values=default_nan_values )
            self.lod=lod


        elif isinstance(lod_mode, (float, int)) or lod_mode==None:
            self.lod=lod

        else:
            raise ValueError("lod_mode must be a path directing to the folder of lod profiles, a numerical value or None.")



    def kmeans_sa(self, ks, feature_set, thresh_k=5, random_state=10 ):
        """Cluster data using a specified feature set with KMeans algorithm and a dictionary of optimal number of clusters to use for each survey (see get_sil_location and get_opt_k functions).

        Args:
            ks (dictionary): Number of clusters (k) or dictionary containing a k for each survey.
            feature_set (list): List of names of features (columns of the ProfileSet.profiles dataframe) to use for clustering.
            thresh_k (int, optional): Minimim k to be used. If survey-specific optimal k is below this value, then k equals the average k of all above threshold values.
            random_state (int, optional): Random seed used to make the randomisation deterministic.

         Returns:
            column (int): new column in ProfileSet.profiles dataframe storing each point cluster label.
        """

        labels_df=kmeans_sa(merged_df=self.profiles,
            ks=ks,
            feature_set=feature_set,
            thresh_k=thresh_k,
            random_state=random_state)

        self.profiles =  self.profiles.merge(labels_df[["point_id","label_k"]], how="right", on="point_id", validate="one_to_one")


    def cleanit(self, l_dicts, cluster_field='label_k', fill_class='sand',
                watermasks_path=None, water_label='water',
                shoremasks_path=None, label_corrections_path=None,
                default_crs={'init': 'epsg:32754'}, crs_dict_string=None,
               geometry_field='coordinates'):
        """Transforms labels k into meaningful classes (sand, water, vegetation ,..) and apply fine-tuning correction, shoremasking and watermasking cleaning procedures.

        Args:
            l_dicts (list): List of classes dictionaries containing the interpretations of each label k in every survey.
            cluster_field (str): Name of the field storing the labels k to transform (default "label_k").
            fill_class (str): Class assigned to points that have no label_k specified in l_dicts.
            watermasks_path (str): Path to the watermasking file.
            water_label: .
            shoremasks_path: Path to the shoremasking file.
            label_corrections_path: Path to the label correction file.
            default_crs: CRS used to digitise correction polygons.
            crs_dict_string: Dictionary storing location codes as key and crs information as values, in dictionary form.
            geometry_field: Field that stores the point geometry (default 'geometry').

         Returns:
            label_k (int): new column in ProfileSet.profiles dataframe storing each point cluster label.
        """


        processes=[]
        if label_corrections_path: processes.append("polygon finetuning")
        if watermasks_path: processes.append("watermasking")
        if shoremasks_path: processes.append("shoremasking")
        if len(processes)==0: processes.append('none')

        self.cleaning_steps = processes

        self.profiles = cleanit(to_clean=self.profiles,l_dicts=l_dicts, cluster_field=cluster_field, fill_class=fill_class,
                    watermasks_path=watermasks_path, water_label=water_label,
                    shoremasks_path=shoremasks_path, label_corrections_path=label_corrections_path,
                    default_crs=default_crs, crs_dict_string=self.crs_dict_string,
                   geometry_field=geometry_field)


class ProfileDynamics():
    """The ProfileDynamics object manages all volumetric and behavioural dynamics computation and plotting."""

    # This dictionary is used to redirect to Pysal classification methods.
    dict_classifiers={'EqualInterval':EqualInterval,
     'FisherJenks':FisherJenks,
     'HeadTailBreaks':HeadTailBreaks,
     'JenksCaspall':JenksCaspall,
     'KClassifiers':KClassifiers,
     'Quantiles':Quantiles,
     'Percentiles':Percentiles,
     'UserDefined':UserDefined}

    def __init__(self, ProfileSet, bins, method, labels=None):
        """Instantiates a ProfileDynamics object with a discretisation scheme for future behavioural dynamics computation.

        Args:
            ProfileSet (object): ProfileSet object that stores all the monitoring information.
            bins (int): Number of bins that both erosional and depositional classes will be partitioned into.
            method (str): Name of the algorithm used to discretise the data. Choose between 'EqualInterval', 'FisherJenks', 'HeadTailBreaks', 'JenksCaspall', 'KClassifiers', 'Quantiles', 'Percentiles'. Head to Pysal GitHub page https://github.com/pysal/mapclassify for more info on classifiers.
            labels (list): List of labels to assign to the bins, in order to be able to name them and keep the analysis meaningful. Name labels from low to high magnitudes, like ["small","medium","high"] in case of bins=3.

        Returns:
            object: ProfileDynamics object.
        """

        if method not in self.dict_classifiers.keys():
            raise NameError(f"{method} not a valid method name. Supported method in this docstring.")

        # Check that bins is valid list, integer or None
        if isinstance(bins, list):
            if len(bins) <= 1:
                raise ValueError("The list provided cannot have zero or only one break point!")
            else:
                if method == 'Percentiles':
                    if min(bins) < 0 or max(bins) > 100:
                        raise ValueError("When using percentiles as break points, the provided list must be in the range [0, 100].")
                    else:
                        print(f"Data will be partitioned using user-provided percentiles.")
                        bins_size=len(bins)
                else:
                    print(f"Data will be partitioned using user-provided bins.")
                    bins_size=len(bins)

        elif isinstance(bins, int):
            if bins<=1:
                raise ValueError("The number of bins to use cannot be zero, one or negative!")
            else:
                print(f"Data will be partitioned into {bins} discrete classes.")
                bins_size=bins

        elif bins==None:
            print("Automatic discretisation will be performed. This method is time consuming with large datasets.")
        else:
            raise ValueError("The bins parameter is not understood. Must be either a list, integer or None.")

        if labels != None:
            if len(labels)!=bins_size:
                raise ValueError("The number of labels doesn't match the provided bins. They must be the same. Alternatively, set labels=None to not use labels.")
            else:
                print("Labels provided.")


        self.bins=bins
        self.method=method
        self.labels=labels
        self.bins_size=bins_size
        self.ProfileSet=ProfileSet


    def save(self, name, out_dir):
        """Save object using pickle.

        Args:
            name (str): Name of the file to save.
            out_dir (str): Path to the directory where to save the object.

        Returns:
            file (pkl): pickle file.
        """

        savetxt=f"{os.path.join(out_dir,name)}.p"
        pickle.dump( self, open( savetxt, "wb" ) )
        print(f"ProfileDynamics object saved in {savetxt} .")

    def compute_multitemporal(self, loc_full, lod_mode='inherited', geometry_column="coordinates", date_field='raw_date', filter_class=None):
        """Compute points elevation change timeseries. Additionally, when LoD profiles had been provided during the ProfileSet object instantiation, compute LoD points as well. Additionally, creates time periods details table (dh_details).

        Args:
            loc_full (dict): Dictionary where keys are location codes ('mar') and values are full lengtht location names ('Marengo').
            lod_mode (str): Values in meters (crs must be projected) used as distance band for neigthours definition in distance weight matrix computation.
            geometry_column (str): Name of the column storing the geometry information.
            date_field (int): Name of the column storing the survey date information.
            filter_class (list, str):  Name or list of names of classes used to filter data. For instance, setting this as 'sand' would compute elevation changes of sand only if 'sand' is a class previously computed.

        Returns:
            dh_df (pd.DataFrame): Dataframe storing elevation change timeseries. Stored in the ProfileDynamics.dh_df attribute.
            dh_details (pd.DataFrame): Dataframe storing information about time periods. Stored in the ProfileDynamics.dh_details attribute.
            lod_dh (pd.DataFrame): Dataframe storing elevation change timeseries of points along LoD transects. Stored in the ProfileDynamics.lod_dh attribute.
        """

        self.dh_df = compute_multitemporal(self.ProfileSet.profiles,
            geometry_column=geometry_column,
            date_field=date_field,
            filter_class=filter_class)

        self.dh_details = create_details_df(self.dh_df, loc_full)
        self.land_limits=pd.DataFrame(self.dh_df.groupby(["location"]).distance.max()).reset_index()

        if lod_mode=='inherited':

            if isinstance(self.ProfileSet.lod, (float, int)):
                lod_df=self.dh_df.groupby(['location','dt']).count().reset_index()
                lod_df['lod']=lod_mode
                self.lod_df=lod_df[['location','dt','lod']]
                self.lod_dh=None
                self.lod_created='yes'

            elif isinstance(self.ProfileSet.lod, pd.DataFrame):
                print("Computing LoD dh data.")
                lod_dh=compute_multitemporal(self.ProfileSet.lod,
                    geometry_column=geometry_column,
                    date_field=date_field,
                    filter_class=None)
                self.lod_df=get_lod_table(lod_dh)
                self.lod_dh=lod_dh
                self.lod_created='yes'

            elif self.ProfileSet.lod==None :
                self.lod_df=None
                self.lod_dh=None
                self.lod_created='no'

            else:
                ValueError("lod_mode attribute of the ProfileSet object not found. Has ProfileSet.extract_profiles been run?")


        elif isinstance(lod_mode, (float, int)):
            lod_df=self.dh_df.groupby(['location','dt']).count().reset_index()
            lod_df['lod']=lod_mode
            self.lod_df=lod_df[['location','dt','lod']]
            self.lod_dh=None

        elif lod_mode==None :
            self.lod_df=None
            self.lod_dh=None
        else:
            raise ValueError("lod_mode must be 'inherited', None or a numeric value.")

    def plot_lod_normality_check(self, locations, alpha=0.05, xlims=None,ylim=None,
        qq_xlims=None,qq_ylims=None,figsize=(7,4)):
        """Plots the error density histograms and Q-Q plot of absolute error along with Saphiro-Wilk and D'Agostino-Pearson tests.

        Args:
            locations (list): Location codes to show plots of.
            alpha (int): p-value threshold to define significance of statistical tests (default=0.05).
            xlims (tuple): Min and max values for the histogram plot x axis.
            ylim (tuple): Min and max values for the histogram plot y axis.
            qq_xlims (tuple): Min and max values for the q-q plot x axis.
            qq_ylims (tuple): Min and max values for the q-q plot y axis.
            figsize (tuple): Width and height (in inches) of the figure.
        """
        if self.lod_created != 'yes':
            raise ValueError("LoD dataset has not been created.\
            Please run the profile extraction again, providing transects specifically placed in pseudo-invariant areas, in order to create the LoD data and statistics.")

        plot_lod_normality_check(multitemp_data=self.lod_dh,
            lod_df=self.lod_df,
            details_table=self.dh_details,
            locations=locations,
            alpha=alpha,
            xlims=xlims,
            ylim=ylim,
            qq_xlims=qq_xlims,
            qq_ylims=qq_ylims,
            figsize=figsize)

    def LISA_site_level(self,
                        mode,
                        distance_value=None,
                        decay=None,
                        k_value=None,
                        geometry_column="geometry"):
        """Performs Hot-Spot analysis using Local Moran's I as Location Indicator of Spatial Association (LISA) with False Discovery Rate (fdr) correction for all the surveys.
            Please refer to PySAL package documentation for more info.

        Args:
            mode (str): If 'distance'(Default), compute spatial weight matrix using a distance-band kernel, specified in distance_value parameter.
                                            If 'knn', spatial weight matrix uses a specified (k_value parameter) of k number closest points to compute weights.
                                            if 'idw', Inverse Distance Weigthing is used with the specified decay power (decay parameter) to compute weight.

            distance_value (int): values in meters (crs must be projected) used as distance band for neigthours definition in distance weight matrix computation.
            decay (int): power of decay to use with IDW.
            k_value (int): number of closest points for neigthours definition in distance weight matrix computation.
            geometry_column (str): field storing the geometry column. If in string form (as loaded from a csv), it will be converted to Point objects. Default='coordinates'.

        Returns:
            hotspots (pd.DataFrame): Dataframe stored in the ProfileDynamics.hotspots attribute, storing the fdr thresholds, local moran-s Is, p and z values and the quadrant
            in which each observation falls in a Moran's scatter plot.
        """

        self.hotspots = LISA_site_level(dh_df=self.dh_df,
                                    mode=mode,
                                    distance_value=distance_value,
                                    geometry_column=geometry_column,
                                    decay=decay,
                                    k_value=k_value,
                                    crs_dict_string=self.ProfileSet.crs_dict_string)


    def discretise(self, lod=None, absolute = True, field="dh", appendix = ("_deposition","_erosion"), print_summary=False):
        """Fit the discretiser (specified when the ProfileDynamics object got instantiated) to the column of interest (default is 'dh').

        Args:
            lod (int, float, pd.DataFrame): Limit of Detections to use. Can be a single value for all surveys, an LoD dataframe or None if no LoD filtering is to be used.
            absolute (bool): whether to discretise the absolute values of the data or not. If True (default), the bins will be derived from absolute values, then, classes will be assigned to both negative and positive numbers accordingly.
            field (str): Name of the column with data to discretise. Data must be numeric.
            appendix (tuple): String to append at the end of each label when absolute is True. Defaults = ("_deposition","_erosion").
            print_summary (bool): If True, prints out the bins specifications and the sum of absolute deviations around class medians as a goodness-metric.

        Returns:
            df_labelled (pd.DataFrame): Dataframe stored in the ProfileDynamics.df_labelled attribute, storing 'markov_tag' column, which holds the class of magnitude of change which will be used in discrete Markov models to compute BCDs.
        """
        df=self.hotspots

        if isinstance(
            lod, (float, int)
        ):  # if a numeric, use this value across all surveys

            full_data_lod=df.copy()
            full_data_lod["lod"]=float(lod)
            full_data_lod[f"{field}_abs"]=full_data_lod.loc[:,field].apply(abs)
            full_data_lod["lod_valid"]=np.where(full_data_lod.loc[:, f"{field}_abs"] < full_data_lod.lod, 'no','yes')
            self.lod_used='yes'
        elif isinstance(lod, pd.DataFrame):
            full_data_lod=pd.merge(df,self.lod_df[['location','dt','lod']], on=['location','dt'], how='left')
            full_data_lod[f"{field}_abs"]=full_data_lod.loc[:,field].apply(abs)
            full_data_lod["lod_valid"]=np.where(full_data_lod.loc[:, f"{field}_abs"] < full_data_lod.lod, 'no','yes')
            self.lod_used='yes'
        else:
            full_data_lod=df.copy()
            full_data_lod["lod"]=['lod_not_used' for i in range(full_data_lod.shape[0])]
            full_data_lod['lod_valid']=['lod_not_used' for i in range(full_data_lod.shape[0])]
            full_data_lod[f"{field}_abs"]=full_data_lod.loc[:,field].apply(abs)
            self.lod_used='no'

        if absolute:
            data_ero=full_data_lod[full_data_lod.loc[:,field] < 0]
            data_depo=full_data_lod[full_data_lod.loc[:,field] > 0]
            data=full_data_lod.loc[:,f"{field}_abs"]
        else:
            data=full_data_lod.loc[:,field]

        bins_discretiser=self.dict_classifiers[self.method](data, self.bins)
        if print_summary==True:
            print(bins_discretiser)
        print(f"\nFit of {self.method} with {self.bins} bins: {bins_discretiser.adcm}")

        bins_depo = list(bins_discretiser.bins[:-1])
        bins_ero = list(np.flip(bins_discretiser.bins[:-1]) * -1)

        bins_ero_JC = UserDefined(data_ero.loc[:,field], bins_ero)
        bins_depo_JC = UserDefined(data_depo.loc[:,field], bins_depo)

        class_erosion=bins_ero_JC.yb.tolist()
        class_deposition=bins_depo_JC.yb.tolist()

        if self.labels == None:
            self.labels=[f"bin_{str(i)}" for i in range(0,self.bins_size)]

        # create dictionaries where keys are the bin number assigned by discretiser and value is the label
        depo_class_names=[i[0]+i[1] for i in list(prod(self.labels,[appendix[0]]))]
        ero_class_names=[i[0]+i[1] for i in list(prod(self.labels,[appendix[1]]))]

        # a general list of tags
        self.tags=[i[0]+i[1] for i in list(prod(self.labels,appendix))]
        reverse_tags=list(reversed(self.tags))
        self.tags_order=[reverse_tags[i] for i in range(1,len(reverse_tags),2)]+[self.tags[i] for i in range(1,len(self.tags),2)]

        states_depo={yb:class_name for yb,class_name in enumerate(depo_class_names)}
        states_ero={yb*-1:class_name for yb,class_name in enumerate(ero_class_names, start=-len(ero_class_names)+1)}

        data_ero["markov_tag"]=[states_ero[i] for i in class_erosion]
        data_depo["markov_tag"]=[states_depo[i] for i in class_deposition]

        self.df_labelled = pd.concat([data_ero,data_depo],ignore_index=False)

    def infer_weights(self, markov_tag_field="markov_tag"):
        """Compute weights from dataset with markov labels to use for e-BCDs computations. The medians of each magnitude class will be used as weight.

        Args:
            markov_tag_field (str): Name of the column where markov tags are stored.

        Returns:
            weights_dict (dict): Dictionary storing magnitude class (markov_tags) weights, stored in the ProfileDynamics.weights_dict attribute.
        """
        joined_re = r"|".join(self.labels)
        self.df_labelled["magnitude_class"] = [re.findall(joined_re,tag)[0] for tag in self.df_labelled.loc[:,markov_tag_field]]

        # create a dictionary with all the medians of the magnitude classes
        class_abs_medians=dict(self.df_labelled.groupby(["magnitude_class"]).dh.apply(lambda x : np.round(np.median(np.abs(x)),2)))

        # create a dictionary with all the weights in each tag
        self.weights_dict={tag:class_abs_medians[re.findall(joined_re,tag)[0]] for tag in self.tags}

    def BCD_compute_location(self,unique_field, mode, store_neg, filterit):
        """It computes first order stochastic transition matrices, Empirical and Residual Beachface Cluster Dynamics (e-BCDS, r-BCDs) indices at the location scale.

        Args:
            unique_field (dataframe): Field storing spatial IDs, which identify a point in spatial dimension but not in temporal dimension. This naturally fits the geometry column so use 'geometry' if in doubt.
            mode (str, float): If a float is provided, this indicates the percentage of time that the points need to be significant clusters across the periods in order to be retained. The no cluster state is renamed "nnn". Insert "drop" to remove all cluster to no-cluster transitions, or 'all' to keep them all and rename cluster-to-no clsuter state 'nnn'.
            store_neg (bool): If True (default), use the subtraction for diminishing trends. Default True.
            filterit (str): if None (default), all points will be used. If 'lod' or 'hotspot', only points above the LoDs or statistically significant clusters (hot/coldspots) will be retained. If 'both', both lod and hotspot filters will be applied.

        Returns:
            (tuple): Tuple containing:
                location_ebcds (pd.DataFrame) Dataframe storing e-BCDs indices and trends (new ProfileDynamics attribute)
                transitions_matrices (dict) (new ProfileDynamics attribute)
                sign(str) can be '-' or '+' for plotting purposes
                markov_details (dict) Dictionary storing number of points (n), number of timesteps (t) and number of transitions (n_transitions) for each location (new ProfileDynamics attribute)
                location_ss (pd.DataFrame) Dataframe storing steady-state vectors and r-BCDs indices (new ProfileDynamics attribute)

        """

        steady_state_victoria = pd.DataFrame()
        markov_indexes_victoria = pd.DataFrame()
        transitions_matrices=dict()

        if filterit == 'lod':
            if self.lod_used == 'no':
                raise ValueError('No LoD was used during the discretise method was run.\
                Please re-run ProfileDynamics.discretise with an LoD value or dataframe to use LoD filter here.')
            else:
                data_to_use=self.df_labelled.query("lod_valid=='yes'")
                print("LoD filter in use. Only points beyond LoDs will be retained.")
        elif filterit == 'hotspot':
            data_to_use=self.df_labelled.query("lisa_q in [1,3]")
            print("Hotspot filter in use. Only statistically significant clusters will be retained.")

        elif filterit == 'both':
            data_to_use=self.df_labelled.query("lod_valid == 'yes' and lisa_q in [1,3]")
            print("Hotspot and LoD filters in use. Only statistically significant clusters beyond LoDs will be retained.")
        else:
            data_to_use=self.df_labelled


        markov_details={}
        for loc in data_to_use.location.unique():

            if "nnn" not in self.tags_order:
                self.tags_order.insert(len(self.tags)//2,"nnn")

            if isinstance(data_to_use.iloc[0][unique_field], shapely.geometry.point.Point):
                data_to_use['spatial_id']=data_to_use.loc[:,unique_field].apply(str)
                unique_field='spatial_id'
            else:
                pass

            dataset_piv = data_to_use.query(f"location=='{loc}'").pivot(
                values="markov_tag", index=unique_field, columns="dt"
            )

            if mode == "all":
                dataset_piv.fillna("nnn", inplace=True)

            elif mode == "drop":
                dataset_piv.dropna(axis="index", how="any", inplace=True)

            elif isinstance(mode, float):

                dts = len(dataset_piv.columns)

                dataset_piv.dropna(axis=0, how="any", thresh=7, inplace=True)
                dataset_piv.fillna("nnn", inplace=True)

            else:
                raise NameError(
                    " Specify the mode ('drop', 'all', or a float number (0.5,1.0,0.95)"
                )



            arr = np.array(dataset_piv)

            m = Markov(arr)

            markov_details.update({loc:{'n':dataset_piv.shape[0],
                                't':dataset_piv.shape[1],
                                'n_transitions':int(m.transitions.sum().sum())}})

            steady_state = pd.DataFrame(m.steady_state, index=m.classes, columns=[loc])
            steady_state_victoria = pd.concat([steady_state, steady_state_victoria], axis=1)

            markov_df = pd.DataFrame(np.round(m.p, 3), index=m.classes, columns=m.classes)

            # if any column is missing from label_order, then add both row and column
            # and populate with zeroes
            if markov_df.columns.all != len(self.tags)+1:  # plus nnn
                # which one is missing?
                missing_states = [
                    state for state in self.tags_order if state not in markov_df.columns
                ]
                for (
                    miss
                ) in missing_states:  # for all missing states, add columns with zeroes
                    markov_df[f"{miss}"] = float(0)
                    # # at the end of the (squared) dataframe
                    last_idx = markov_df.shape[0]
                    markov_df.loc[last_idx + 1] = [float(0) for i in markov_df.columns]

                # get a list of str of the missing states
                to_rename = markov_df.index.to_list()[-len(missing_states) :]
                for i, j in zip(to_rename, missing_states):
                    markov_df.rename({i: j}, inplace=True)

            markov_ordered = markov_df.loc[self.tags_order, self.tags_order]

            # When no transition occurred, replace NaN with a 0.
            markov_ordered.fillna(0, inplace=True)

            idx_matrix=len(self.tags)//2

            dd = markov_ordered.iloc[:idx_matrix, :idx_matrix]
            dd = dd[dd.columns[::-1]]

            ee = markov_ordered.iloc[idx_matrix+1:, idx_matrix+1:]
            ee = ee.reindex(index=ee.index[::-1])

            de = markov_ordered.iloc[:idx_matrix, idx_matrix+1:]

            ed = markov_ordered.iloc[idx_matrix+1:, :idx_matrix]
            ed = ed.reindex(index=ed.index[::-1])
            ed = ed[ed.columns[::-1]]

            list_markovs = [ee, ed, dd, de]
            transitions_matrices.update({loc:list_markovs})
            dict_markovs = {"ee": ee, "ed": ed, "dd": dd, "de": de}


            # create index dataframe

            for arr_markov in dict_markovs.keys():

                idx, trend, sign = get_coastal_Markov(
                    dict_markovs[arr_markov], weights_dict=self.weights_dict, store_neg=store_neg
                )

                idx_coastal_markov_dict = {
                    "location": loc,
                    "sub_matrix": arr_markov,
                    "coastal_markov_idx": idx,
                    "trend": trend,
                    "sign": sign,
                }

                idx_coastal_markov_df = pd.DataFrame(idx_coastal_markov_dict, index=[0])
                markov_indexes_victoria = pd.concat(
                    [idx_coastal_markov_df, markov_indexes_victoria], ignore_index=True
                )


            # here I translate the submatrix codes into the names and store them in a column

            titles=["Erosional", "Recovery", "Depositional","Vulnerability"]
            translation=dict(zip(["ee","ed","dd","de"],titles))
            markov_indexes_victoria["states_labels"]=markov_indexes_victoria["sub_matrix"].map(translation)

            self.location_ebcds=markov_indexes_victoria
            self.transitions_matrices=transitions_matrices
            self.markov_details=markov_details


            ss_victoria_ordered=steady_state_victoria.loc[self.tags_order,:]
            ss_victoria_ordered.drop("nnn",inplace=True)

            # Create erosion and deposition sub-matrix
            erosion=ss_victoria_ordered.iloc[idx_matrix:,:].transpose()
            erosion["total_erosion"]=erosion.sum(axis=1)
            erosion=erosion.reset_index().rename({"index":"location"},axis=1)

            deposition=ss_victoria_ordered.iloc[:-idx_matrix,:].transpose()
            deposition["total_deposition"]=deposition.sum(axis=1)
            deposition=deposition.reset_index().rename({"index":"location"},axis=1)

            merged=pd.concat([deposition,erosion.iloc[:,1:]], axis=1)
            merged["r_bcds"]=(merged.total_deposition- merged.total_erosion)*100
            merged=merged.set_index("location").transpose().rename_axis('', axis=1)

            self.location_ss=merged

    def BCD_compute_transects(self, loc_specs, reliable_action, dirNameTrans):
        """It computes Residual Beachface Cluster Dynamics (r-BCDs) indices at the transect scale.

        Args:
            loc_specs (dict): Dictionary specifying minimum required valid points per transect and minimum number of timesteps required to consider a transect reliable. Must be provided in this neste dictionary form: loc_specs={'mar':{'thresh':6,'min_points':6}, 'leo': {'thresh':4, 'min_points':5}}
            reliable_action (str): If 'drop', only retains transects that are reliable. Otherwise, keep all the transects. The transect reliability depends on the thresh and min_points parameters provided with loc_specs dictionary.
            dirNameTrans (str): Path of the directory containing the transect files (.gpkg, .shp).

        Returns:
            transects_rbcd (gpd.GeoDataFrame): GeoDataframe storing r-BCDs indices (new ProfileDynamics attribute).
        """
        self.transects_rbcd=get_rbcd_transect(df_labelled=self.df_labelled,
                  loc_specs=loc_specs,
                  dirNameTrans=dirNameTrans,
                  reliable_action=reliable_action,
                  labels_order=self.tags_order,
                  loc_codes=self.ProfileSet.loc_codes,
                  crs_dict_string=self.ProfileSet.crs_dict_string)


    def compute_volumetrics(self, lod, outliers=False, sigma_n=3):
        """Generate volumetric and altimetric change statistics both at the transect and location levels.

        Args:
            lod (pd.DataFrame, float, bool): If a DataFRame is provided, it must be have a column named location with location codes and another called lod with lod values. If a float is provided, uses this value across all the serveys. If False is provided, then don't use LoD filters.
            outliers (bool): When True, use the specified number of standard deviations (sigma_n) to exclude outliers. If False, retain all the points.
            sigma_n (int): Number of standard deviation to use to exclude outliers (default=3).

        Returns:
            ProfileDynamics attributes: Two new attributes named location_volumetrics (pd.DataFrame) and transects_volumetrics (pd.DataFrame), which provide change information at the location and single transect scales respectively.
        """
        self.dh_df["date_pre_dt"]=[datetime.datetime.strptime(str(pre),'%Y%m%d') for pre in self.dh_df.date_pre]
        self.dh_df["date_post_dt"]=[datetime.datetime.strptime(str(post),'%Y%m%d') for post in self.dh_df.date_post]

        if isinstance(lod , pd.DataFrame):
            print("Using LoDs.")
            self.lod_used='yes'
        elif isinstance(lod , (int,float)):
            print("Using LoDs.")
            self.lod_used='yes'
        else:
            print("No LoD is used.")
            self.lod_used='no'

        self.location_volumetrics = get_state_vol_table(self.dh_df, lod=lod,
                                              full_specs_table=self.dh_details, dx=self.ProfileSet.sampling_step)

        self.transects_volumetrics = get_transects_vol_table(self.dh_df, lod=lod,
                                        transect_spacing=self.ProfileSet.transects_spacing,
                                        dx=self.ProfileSet.sampling_step,
                                        outliers=outliers,sigma_n=sigma_n,
                                        full_specs_table=self.dh_details)

    def plot_transects(self, location, tr_id, classified, dt=None, from_date=None, to_date=None, details_df=None,
                       figsize=None, palette ={"sand": "r", "water": "b", "veg": "g", "no_sand": "k"}):
        """Visualise elevation profile changes between pre and post dates in selected locations, transect ids and time periods. If the data has been classified, it can also see what class each point was classified into.

        Warning:
            if classified=False, the points that have been filtered out will not be displayed. If classified=True, then all points will be plotted. This is done to have a visual understanding of the cleaning procedure applied and appreciate what is actually retained to compute change statistics.

        Args:
            location (str): Location code of of the transect to visualise.
            tr_id (int): Id of the transect to visualise.
            classified (bool): If True, colour the points (set palette argument to control the colors) based on class.
            dt (list): List of time periods (dt) to plot, like ['dt_0', 'dt_4', 'dt_2'].
            from_date (str): Date of the pre-survey in raw format (yyyymmdd).
            to_date (str): Date of the pre-survey (yyyymmdd).
            figsize (tuple): Width and height (in inches) of the figure.
            palette (dict): Dictionary where keys are class names and values the assigned colors in matplotlib-understandable format, like {"sand": "r", "water": "b", "veg": "g", "no_sand": "k"}.
        """
        if dt != None:

            if isinstance(dt, list):
                periods=dt
            else:
                periods=[dt]

            for dt_i in periods:
                f,ax=plt.subplots(1, figsize=(10,5))

                details=self.dh_details.query(f"location=='{location}' and dt == '{dt_i}'")
                full_loc=details.iloc[0]["loc_full"]
                from_date=details.iloc[0]["date_pre"]
                to_date=details.iloc[0]["date_post"]


                if classified == False:
                    data=self.dh_df.query(f"location=='{location}' and tr_id=={tr_id} and dt=='{dt_i}'")

                    sb.scatterplot(data=data, x="distance", y='z_pre', color='b', size=5)
                    sb.lineplot(data=data,x="distance",y='z_pre', color="b", label='Pre')

                    sb.scatterplot(data=data, x="distance", y='z_post', color='r', size=5, legend=False)
                    sb.lineplot(data=data,x="distance",y='z_post', color="r",ls='--', label='Post')

                elif classified:
                    from_classed=self.ProfileSet.profiles.query(f"location=='{location}' and tr_id=={tr_id} and raw_date=={from_date}")
                    to_classed=self.ProfileSet.profiles.query(f"location=='{location}' and tr_id=={tr_id} and raw_date=={to_date}")

                    sb.scatterplot(data=from_classed, x="distance", y='z', s=20, alpha=1, hue='pt_class',palette=palette)
                    sb.lineplot(data=from_classed,x="distance", y='z', color='k', alpha=.5,label='Pre', linewidth=2)


                    sb.scatterplot(data=to_classed, x="distance", y='z', s=20, alpha=1, hue='pt_class', legend=False, palette=palette)
                    sb.lineplot(data=to_classed,x="distance",y='z', color='r', alpha=.5, linewidth=2, label='Post')
                else:
                    raise ValueError("The 'classified' parameter must be either True or False.")



                ax.set_ylabel('Elevation (m)')
                ax.set_xlabel('Distance alongshore (m)')
                ax.set_title(f"Location: {full_loc}\nTransect: {tr_id}\nFrom {from_date} to {to_date} ({dt_i})");

        elif dt == None:

            if from_date != None and to_date != None:
                f,ax=plt.subplots(1, figsize=(10,5))

                full_loc=self.dh_details.query(f"location=='{location}'").iloc[0]["loc_full"]

                if classified == False:
                    data_pre=self.dh_df.query(f"location=='{location}' and tr_id=={tr_id} and date_pre == {from_date}")
                    data_post=self.dh_df.query(f"location=='{location}' and tr_id=={tr_id} and date_post == {to_date}")

                    sb.scatterplot(data=data_pre, x="distance", y='z_pre', color='b', size=5)
                    sb.lineplot(data=data_pre,x="distance",y='z_pre', color="b", label='Pre')

                    sb.scatterplot(data=data_post, x="distance", y='z_post', color='r', size=5, legend=False)
                    sb.lineplot(data=data_post,x="distance",y='z_post', color="r",ls='--', label='Post')

                elif classified:

                    from_classed=self.ProfileSet.profiles.query(f"location=='{location}' and tr_id=={tr_id} and raw_date=={from_date}")
                    to_classed=self.ProfileSet.profiles.query(f"location=='{location}' and tr_id=={tr_id} and raw_date=={to_date}")

                    sb.scatterplot(data=from_classed, x="distance", y='z', s=20, alpha=1, hue='pt_class',palette=palette)
                    sb.lineplot(data=from_classed,x="distance", y='z', color='k', alpha=.5,label='Pre', linewidth=2)


                    sb.scatterplot(data=to_classed, x="distance", y='z', s=20, alpha=1, hue='pt_class', legend=False, palette=palette)
                    sb.lineplot(data=to_classed,x="distance",y='z', color='r', alpha=.5, linewidth=2, label='Post')

                ax.set_ylabel('Elevation (m)')
                ax.set_xlabel('Distance alongshore (m)')
                ax.set_title(f"Location: {full_loc}\nTransect: {tr_id}\nFrom {from_date} to {to_date}");

            else:
                raise ValueError("Only one of date_from/date_to dates has been provided. Please provide both dates or use the dt parameter only.")


    def plot_transect_mecs(self, location, tr_id, lod=None, figsize=(10,5)):
        """ Barplots summarising MEC of each transect across time and regressionplot.

        Args:
            location (str): Location code of of the transect to visualise.
            tr_id (int): Id of the transect to visualise.
            lod (pd.DataFrame): If provided, filter dataset using survey-specific LoDs.
            figsize (tuple): Width and height (in inches) of the figure.
        """

        details=self.dh_details.query(f"location=='{location}'")
        full_loc=details.iloc[0]["loc_full"]

        f,axs=plt.subplots(nrows=1,
                          ncols=2,
                          figsize=figsize)

        data=self.dh_df.query(f"location=='{location}' and tr_id=={tr_id}")
        filter_used=self.dh_df.class_filter.unique()[0]


        if isinstance(lod, pd.DataFrame):
            print("Using LoDs.")
            lods=self.lod_df.query(f"location=='{location}'")[["dt","lod"]]
            data=pd.merge(data,lods, how='left', on='dt')
            data["dh_abs"]=[abs(dh_i) for dh_i in data.dh]
            data["dh"]=np.where(data.dh_abs<data.lod, 0,data.dh)


        if filter_used != "no_filters_applied":
            print(f"Statistics are computed based on the following classes only: {filter_used}")

        mecs=data.groupby(['dt']).dh.sum()/data.groupby(['dt']).geometry.count()
        mecs=mecs.reset_index()
        mecs.columns=['dt','mec']
        mecs['dt_i']=[int(mecs.iloc[i,0].split("_")[-1]) for i in range(mecs.dt.shape[0])]
        mecs['net_change']=['positive' if mec_i >=0 else 'negative' for mec_i in mecs.mec]


        barplot=sb.barplot(data=mecs.sort_values(['dt_i']),x='mec',y='dt', ax=axs[0],
                           hue='net_change', palette={'positive':'b','negative': 'r'})

        barplot.set_ylabel("Time period (dt)")
        barplot.set_xlabel("Mean Elevation Change (m)")
        barplot.set_title('Barplot')

        trend=sb.regplot(data=mecs, x='dt_i',y='mec',ax=axs[1])
        sb.lineplot(data=mecs, x='dt',y='mec',ax=axs[1])
        trend.set_title('Trend')

        if 'dh_abs' in data.columns:
            data.drop('dh_abs', axis=1, inplace=True)

        f.suptitle(f"Location: {full_loc}\nTransect: {tr_id}");


    def plot_single_loc(self,
                        loc_subset,
                        colors_dict,
                        figsize=(8.5,4),
                        linewidth=1.5,
                        out_date_format="%d/%m/%Y",
                        xlabel="Survey date" ,
                        ylabel="Volume change (m³)",
                        suptitle="Volumetric Timeseries"):
        """ Volume change timeseries of a location. Solid line are period-specific change and dashed ones are the cumulative change from the start of the monitoring.

        Args:
            loc_subset (str): Location code of of the transect to visualise.
            colors_dict (int): Id of the transect to visualise.
            figsize (tuple): Width and height (in inches) of the figure.
            linewidth (float): Width of the lines.
            out_date_format (str): Format of the output date. Must be in the datetime.strptime format, like "%d/%m/%Y".
            xlabel (str): Label of the x axis.
            ylabel (str): Label of the y axis.
            suptitle (str): Title of the plot.
        """

        plot_single_loc(self.location_volumetrics,
                        loc_subset=loc_subset,
                        figsize=figsize,
                        colors_dict=colors_dict,
                        linewidth=linewidth,
                        out_date_format=out_date_format,
                        xlabel=xlabel,
                        ylabel=ylabel,
                        suptitle=suptitle)

    def plot_alongshore_change(self,
        mode,
        lod,
        location_subset=None,
        dt_subset=None,
        ax2_y_lims=(-8, 5),
        save=False,
        save_path=None,
        dpi=300,
        img_type=".png",
        from_land=True,
        from_origin=True,
        fig_size=None,
        font_scale=1,
        plots_spacing=0,
        bottom=False,
        y_heat_bottom_limit=None,
        heat_yticklabels_freq=None,
        heat_xticklabels_freq=None,
        outliers=False,
        sigma_n=3):
        """ Display longshore altimetric and volumetric beach changes plots. A subset of locations and periods can be plotted. If LoD parameter is True (default), then white cells in the altimetric heatmap are values within LoD. Grey cells is no data or filtered-out points.

        Args:
            mode (str): If 'subset', only a subset of locations and dts are plotted. If 'all', all periods and locations are plotted. .
            lod (pd.DataFrame, float, bool): If a DataFRame is provided, it must be have a column named location with location codes and another called lod with lod values. If a float is provided, uses this value across all the serveys. If False is provided, then don't use LoD filters.
            location_subset (list): List of strings containing the location codes (e.g. wbl) to be plotted.
            dt_subset (list): List of strings containing the period codes (e.g. dt_0) to be plotted.
            ax2_y_lims (tuple): Limits of y-axis of alonghsore volumetric change plot. Default is (-8,5).
            save (bool): If True, saves the plots in the specified save_path. False is default.
            save_path (path): Full path to a folder (e.g. C:\\preferred\\folder\\) where to save plots.
            dpi (int): Resolution in Dot Per Inch (DPI) to save the images.
            img_type (str): '.png','.pdf', '.ps', '.svg'. Format of the saved figures.
            from_land (bool): If True (default), cross-shore distances are transformed into landward distances, where 0 is the end of beachface.
            from_origin (bool): If True (default), transect IDS are transformed in alongshore distance from origin (tr_id=0). It requires regularly spaced transects.
            fig_size (tuple): Width and height (in inches) of the figure.
            font_scale (float): Scale of text. Default=1.
            plots_spacing (float): Vertical spacing of the heatmap and alongshore change plots. Default = 0.
            bottom (bool): If True (default), rows are extended seaward too, up to y_heat_bottom_limit. If False, only distances from 0 to the first valid values will be added.
            y_heat_bottom_limit (int): Lower boundary distance (seaward) to extend all transects to.
            heat_yticklabels_freq (int): Plot a labels every n labels in the heatmap y axis.
            heat_xticklabels_freq (int): Plot a labels every n labels in the heatmap x axis.
            outliers (bool): When True, use the specified number of standard deviation to exclude outliers. If False, retain all the points.
            sigma_n (int): Number of standard deviation to use to exclude outliers (default=3).
        """

        plot_alongshore_change(self.dh_df,
            mode=mode,
            lod=lod,
            full_specs_table=self.dh_details,
            return_data=False,
            location_subset=location_subset,
            dt_subset=dt_subset,
            ax2_y_lims=ax2_y_lims,
            save=save,
            save_path=save_path,
            dpi=dpi,
            img_type=img_type,
            from_land=from_land,
            from_origin=from_origin,
            add_orient=False,
            fig_size=fig_size,
            font_scale=font_scale,
            plots_spacing=plots_spacing,
            bottom=bottom,
            y_heat_bottom_limit=y_heat_bottom_limit,
            transect_spacing=self.ProfileSet.transects_spacing,
            along_transect_sampling_step=self.ProfileSet.sampling_step,
            heat_yticklabels_freq=heat_yticklabels_freq,
            heat_xticklabels_freq=heat_xticklabels_freq,
            outliers=outliers,
            sigma_n=sigma_n)


    def plot_mec_evolution(self,
        loc_order=None,
        date_format="%d.%m.%y",
        scale_mode="equal",
        x_diff=None,
        dates_step=50,
        x_limits=(-0.41, 0.41),
        x_binning=5,
        figure_size=(7, 4),
        font_scale=0.75,
        dpi=300,
        img_type=".png",
        save_fig=False,
        name_fig=f"Mean Elevation Changes",
        save_path=None):
        """Plot all locations period-specific and cumulative MECs side by side in a comparable way.

        Args:
            loc_order (list): List of location codes specifying the order (left-to-right) to plot timeseries.
            date_format (str): Format of the output date. Must be in the datetime.strptime format, like "%d.%m.%y".
            scale_mode (str): If ""equal", each location x axis limits are the same. This is useful for reliable visual comparison of MEC. If "auto", each location will have different x limits optimised for each location.
            x_diff (dict): A dictionary where keys are location codes and values are x_min and x_max in a list, like {'leo': [-0.12, 0.1]}. If provided, this allows to keep all the locations x axis equal, except the locations specified in this dictionary.
            dates_step (int): Plot one date label ever n in the y-axis.
            x_limits (tuple): A tuple containing x_min and x_max to apply to all the locations x axis (except those specified with x_diff, if any).
            x_binning (int): Plot one label ever n in the locations x-axis.
            figure_size (tuple): Width and height (in inches) of the figure.
            font_scale (float): Scale of text. Default=1.
            dpi (int): Resolution in Dot Per Inch (DPI) to save the images.
            img_type (str): '.png','.pdf', '.ps', '.svg'. Format of the saved figures.
            save_fig (bool): If True, saves the plots in the specified save_path. False is default.
            name_fig (str): Name to give to the figure to be saved.
            save_path (str): Full path to a folder, like r"C:\\preferred\\folder\\", where to save the figure.
        """


        plot_mec_evolution(self.location_volumetrics,
            location_field='location',
            loc_order=loc_order,
            date_from_field="date_from",
            date_to_field="date_to",
            date_format=date_format,
            scale_mode=scale_mode,
            x_diff=x_diff,
            dates_step=dates_step,
            x_limits=x_limits,
            x_binning=x_binning,
            figure_size=figure_size,
            font_scale=font_scale,
            dpi=dpi,
            img_type=img_type,
            save_fig=save_fig,
            name_fig=name_fig,
            save_path=save_path)


    def plot_trans_matrices(self,
                        relabel_dict=None,
                        titles = ["Erosional", "Recovery", "Depositional", "Vulnerability"],
                        cmaps = ["Reds", "Greens", "Blues", "PuRd"],
                        fig_size=(6, 4),
                        heat_annot_size=10,
                        font_scale=0.75,
                        dpi=300,
                        save_it=False,
                        save_output=r"C:\\your\\preferred\\folder\\"):
        """Plot location-specific first-order stochastic matrices, representing the probability of each transition of a point (discretised into magnitude of change classes) of going from an initial state (matrices row) to a second state (matrices column) in one time period. These probabilities are the absis for the computation of the e-BCDs

        Args:
            relabel_dict (dict): Dictionary where keys are the magnitude of change classes and values a corresponding abbreviation, like {'Undefined_deposition': 'ue','Undefined_erosion': 'ue', 'Small_deposition': 'sd'}. This is useful in case of long names and abbreviations allows a much better visualisation.
            titles (list): List of names to assign to each submatrices, which define the behaviour of each matrices transitions. Default is ["Erosional", "Recovery", "Depositional", "Vulnerability"].
            cmaps (list) : List of color ramps to assign to each matrices. It must be in the same order as specified in the titles parameter. Default is ["Reds", "Greens", "Blues", "PuRd"].
            fig_size (tuple): Width and height (in inches) of the figure.
            heat_annot_size (int): Size of the annotations (probabilities) inside each matrices cells.
            font_scale (float): Scale of text. Default=1.
            dpi (int): Resolution in Dot Per Inch (DPI) to save the images.
            save_it (bool): If True, saves the plots in the specified save_path. False is default.
            save_output (str): Full path to a folder, like r"C:\\preferred\\folder\\", where to save the figure.
        """

        for loc in self.df_labelled.location.unique():
            std_excluding_nnn = np.array(self.transitions_matrices[f"{loc}"]).flatten().std()
            exclude_outliers = np.round(3 * std_excluding_nnn, 1)

            f, axs = plt.subplots(nrows=2, ncols=2, figsize=fig_size)

            for ax_i, heat, title, cmap_i in zip(
                axs.flatten(), self.transitions_matrices[f"{loc}"], titles, cmaps
            ):
                if isinstance(relabel_dict,dict):
                    heat=heat.rename(columns=relabel_dict, index=relabel_dict)
                else:
                    pass

                sb.heatmap(
                    heat,
                    cmap=cmap_i,
                    annot=True,
                    linewidths=1,
                    linecolor="white",
                    vmin=0,
                    vmax=exclude_outliers,
                    annot_kws={"size": heat_annot_size},
                    ax=ax_i,
                )
                ax_i.set_title(f"{title}", size=9)
                title = f.suptitle(f"{loc} (n={self.markov_details[loc]['n']}, t={self.markov_details[loc]['t']}, trans:{int(self.markov_details[loc]['n_transitions'])}) ")
                title.set_position([0.5, 1.03])
                f.tight_layout(pad=1)

            if bool(save_it):
                f.savefig(f"{save_output}{loc}_sub_matrices_.png", dpi=dpi)
            else:
                pass


    def plot_location_ebcds(self,
                        loc_order,
                        palette_dyn_states ={"Erosional":"r","Recovery":"g","Depositional":"b","Vulnerability":"purple"},
                        b_order=["Erosional","Recovery","Depositional","Vulnerability"],
                        xticks_labels=None,
                        figsize=(8,4),
                        font_scale=0.8):
        """Plot location-specific Empirical Beachface Cluster Dynamics indices and their trend in the form as a plus sign for increasing trend and a minus for decreasing trends.

        Args:
            loc_order (list): List of location codes specifying the order (left-to-right) to plot.
            palette_dyn_states (dict): Dictionary where keys are the behaviours and values the color to be assigned, like {"Erosional":"r","Recovery":"g","Depositional":"b","Vulnerability":"purple"}.
            b_order (list) : List of behaviours specifying the order (left-to-right) to plot. Default is ["Erosional","Recovery","Depositional","Vulnerability"].
            figsize (tuple): Width and height (in inches) of the figure.
            xticks_labels (list): If provided, rename the x labels of the plot (locations) from left to right, like ["Marengo","St. Leo"].
            font_scale (float): Scale of text. Default=0.8.
        """

        f,ax=plt.subplots(figsize=figsize)

        sb.set_context("paper", font_scale=font_scale)
        sb.set_style("whitegrid")

        plot_bars=sb.barplot(data=self.location_ebcds,x="location", y=self.location_ebcds.coastal_markov_idx,hue="states_labels", hue_order=b_order,
                  order=loc_order, palette=palette_dyn_states)


        if xticks_labels:
            ax.set_xticklabels(labels=xticks_labels)
        ax.set_xlabel("")
        ax.set_ylabel("e-BCD")

        txt_heights=[i.get_height() for i in ax.patches]

        signs=[]
        for i in b_order:
            for j in loc_order:
                sign=self.location_ebcds.query(f"location == '{j}' & states_labels=='{i}'").sign.values
                signs.append(sign)


        for p,txt_height,sign in zip(ax.patches,txt_heights,signs):
            width, height = p.get_width(), p.get_height()
            x, y = p.get_xy()
            ax.text(x+width/2,
                 txt_height - 0.095,
                 sign[0],
                 horizontalalignment='center',
                 verticalalignment='center',
                    color="white",
                   fontsize=13,
                   fontweight='heavy')

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles=handles[:], labels=labels[:], loc=0)

        plt.tight_layout()
