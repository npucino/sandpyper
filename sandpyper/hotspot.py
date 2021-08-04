"""Hotspot module."""
import glob
import warnings

from tqdm.notebook import tqdm
import pandas as pd
import shapely
from shapely.geometry import Point
from shapely import wkt
import geopandas as gpd
import numpy as np
from pysal.lib import weights
import pysal.explore.esda.moran as moran
from pysal.explore.esda.util import fdr
from sandpyper.outils import coords_to_points, getListOfFiles, getLoc, spatial_id, create_details_df
from sandpyper.dynamics import get_coastal_Markov,  compute_multitemporal, get_lod_table, plot_lod_normality_check, get_rbcd_transect
from sandpyper.volumetrics import (get_state_vol_table, get_transects_vol_table,
                                   plot_alongshore_change, plot_mec_evolution, plot_single_loc)
from itertools import product as prod
from pysal.viz.mapclassify import (EqualInterval,
                                   FisherJenks,
                                   HeadTailBreaks,
                                   JenksCaspall,
                                   KClassifiers,
                                   Quantiles,
                                   Percentiles,
                                   UserDefined)


from pysal.explore.giddy.markov import Markov
import matplotlib.pyplot as plt
import seaborn as sb
import re
import datetime
import pickle
import os


class ProfileDynamics():
    """
    Create a Discretiser instance that classify a numeric field of a dataframe (with its fit method) into bins, using a specific method.

    Args:
        bins (list, int or None): If a list is provided, use those as break points to discretise the data.
        If an integer is provided, this defines the desired number of bins. If None (Default), it automatically finds the best number of bins.
        method (str): Name of the discretisation method to use, as specified by Pysal. Must be one of:

        EqualInterval
        FisherJenks
        HeadTailBreaks
        JenksCaspall
        KClassifiers
        Quantiles
        Percentiles

        Please visit Pysal documentation for more information :
        https://pysal.org/notebooks/viz/mapclassify/intro.html#Map-Classifiers-Supported

    returns:
        Discretiser class.
    """

    dict_classifiers={'EqualInterval':EqualInterval,
     'FisherJenks':FisherJenks,
     'HeadTailBreaks':HeadTailBreaks,
     'JenksCaspall':JenksCaspall,
     'KClassifiers':KClassifiers,
     'Quantiles':Quantiles,
     'Percentiles':Percentiles,
     'UserDefined':UserDefined}

    def __init__(self, ProfileSet, bins, method, labels=None):

        if method not in self.dict_classifiers.keys():
            raise NameError(f"{method} not a valid method name. Supported method in this docstring.")

        # Check that bins is valid list, integer or None
        if isinstance(bins, list):
            if len(bins) <= 1:
                raise ValueError("The list provided cannot have zero or only one break point!")
            else:
                if method == 'Percentiles':
                    if min(bins) < 0 or max(bins) > 100:
                        raise ValueError("When using precentiles as break points, the provided list must be in the range [0, 100].")
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
        savetxt=f"{os.path.join(out_dir,name)}.p"
        pickle.dump( self, open( savetxt, "wb" ) )
        print(f"ProfileDynamics object saved in {savetxt} .")

    def compute_multitemporal(self, loc_full, lod_mode='inherited', geometry_column="coordinates", date_field='raw_date', filter_class=None):
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

    def plot_lod_normality_check(self, locations, details_table=None, lod_df=None, alpha=0.05,xlims=None,ylim=None,qq_xlims=None,qq_ylims=None,figsize=(7,4)):

        if self.lod_created != 'yes':
            raise ValueError("LoD dataset has not been created.\
            Please run the profile extraction agains, providing transects specifically placed in pseudo-invariant areas, in order to create the LoD data and statistics.")

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
                        distance_value,
                        geometry_column="geometry"):

        self.hotspots = LISA_site_level(self.dh_df,
                                    mode=mode,
                                    distance_value=distance_value,
                                    geometry_column=geometry_column,
                                    crs_dict_string=self.ProfileSet.crs_dict_string)

    def discretise(self, lod=None, absolute = True, field="dh", appendix = ("_deposition","_erosion"), print_summary=False):
        """
        Fit discretiser to the dataframe containing the field of interest.

        Args:
            df (Pandas DataFrame): Dataframe with colum to discretise.
            absolute (bool): wether to discretise the absolute values of the data or not. If True (default), the bins will
            be derived from absolute values, then, classes will be assigned to both negative and positive numbers accordingly.
            field (str): Name of the column with data to discretise. Data must be numeric.
            appendix (tuple): String to append at the end of each label when absolute is True. Defaults = ("_deposition","_erosion").

        returns:
            Input dataframe with added columns containing the bins and a column with the labels (if provided).
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
        """Compute weights from dataset with markov labels to use for e-BCDs computation.
            The medians of each magnitude class will be used as weight.

        Args:
            data (Pandas Dataframe): Pandas dataframe.
            markov_tag_field (str): Name of the column where markov tags are stored.

        Returns:
            dict, containing the markov tags and associated weights.
        """
        joined_re = r"|".join(self.labels)
        self.df_labelled["magnitude_class"] = [re.findall(joined_re,tag)[0] for tag in self.df_labelled.loc[:,markov_tag_field]]

        # create a dictionary with all the medians of the magnitude classes
        class_abs_medians=dict(self.df_labelled.groupby(["magnitude_class"]).dh.apply(lambda x : np.round(np.median(np.abs(x)),2)))

        # create a dictionary with all the weights in each tag
        self.weights_dict={tag:class_abs_medians[re.findall(joined_re,tag)[0]] for tag in self.tags}

    def BCD_compute_location(self,unique_field, mode,store_neg, filterit):
        '''filter (str): if None (default), all points will be used.
        If 'lod' or 'hotspot', only points above the LoDs or statistically significant clusters (hot/coldspots)
        will be retained. If 'both', both lod and hotspot filters will be applied.'''

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
                thrsh = int(mode * dts)

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

    def BCD_compute_transects(self, thresh, min_points, reliable_action, dirNameTrans=None):


        self.transects_rbcd=get_rbcd_transect(df_labelled=self.df_labelled,
                  thresh=thresh, min_points=min_points,
                  dirNameTrans=self.ProfileSet.dirNameTrans,
                  reliable_action=reliable_action,
                  labels_order=self.tags_order,
                  loc_codes=self.ProfileSet.loc_codes,
                  crs_dict_string=self.ProfileSet.crs_dict_string)


    def compute_volumetrics(self, lod, outliers=False, sigma_n=3):

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
                        from_classed=self.ProfileSet.profiles.query(f"location=='{location}' and tr_id=={tr_id} and raw_date=='{str(from_date)}'")
                        to_classed=self.ProfileSet.profiles.query(f"location=='{location}' and tr_id=={tr_id} and raw_date=='{str(to_date)}'")

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
                        data_pre=self.dh_df.query(f"location=='{location}' and tr_id=={tr_id} and date_pre == '{from_date}'")
                        data_post=self.dh_df.query(f"location=='{location}' and tr_id=={tr_id} and date_post == '{to_date}'")

                        sb.scatterplot(data=data_pre, x="distance", y='z_pre', color='b', size=5)
                        sb.lineplot(data=data_pre,x="distance",y='z_pre', color="b", label='Pre')

                        sb.scatterplot(data=data_post, x="distance", y='z_post', color='r', size=5, legend=False)
                        sb.lineplot(data=data_post,x="distance",y='z_post', color="r",ls='--', label='Post')

                    elif classified:

                        from_classed=self.ProfileSet.profiles.query(f"location=='{location}' and tr_id=={tr_id} and raw_date=='{str(from_date)}'")
                        to_classed=self.ProfileSet.profiles.query(f"location=='{location}' and tr_id=={tr_id} and raw_date=='{str(to_date)}'")

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
        full_specs_table=None,
        return_data=False,
        location_subset=None,
        dt_subset=None,
        ax2_y_lims=(-8, 5),
        save=False,
        save_path=None,
        dpi=300,
        img_type=".png",
        from_land=True,
        from_origin=True,
        add_orient=False,
        fig_size=None,
        font_scale=1,
        plots_spacing=0,
        bottom=False,
        y_heat_bottom_limit=None,
        transect_spacing=None,
        heat_yticklabels_freq=None,
        heat_xticklabels_freq=None,
        outliers=False,
        sigma_n=3):


        plot_alongshore_change(self.dh_df,
            mode=mode,
            lod=lod,
            full_specs_table=self.dh_details,
            return_data=return_data,
            location_subset=location_subset,
            dt_subset=dt_subset,
            ax2_y_lims=ax2_y_lims,
            save=save,
            save_path=save_path,
            dpi=dpi,
            img_type=img_type,
            from_land=from_land,
            from_origin=from_origin,
            add_orient=add_orient,
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
        location_field,
        loc_order,
        date_from_field="date_from",
        date_to_field="date_to",
        date_format="%d.%m.%y",
        scale_mode="equal",
        x_diff=None,
        dates_step=50,
        x_limits=(-0.41, 0.41),
        x_binning=5,
        figure_size=(7, 4),
        font_scale=0.75,
        sort_locations=True,
        dpi=300,
        img_type=".png",
        save_fig=False,
        name_fig=f"Mean Elevation Changes",
        save_path=None):


        plot_mec_evolution(self.location_volumetrics,
            location_field=location_field,
            loc_order=loc_order,
            date_from_field=date_from_field,
            date_to_field=date_to_field,
            date_format=date_format,
            scale_mode=scale_mode,
            x_diff=x_diff,
            dates_step=dates_step,
            x_limits=x_limits,
            x_binning=x_binning,
            figure_size=figure_size,
            font_scale=font_scale,
            sort_locations=sort_locations,
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
                        save_output="C:\\your\\preferred\\folder\\"):

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
                        loc_order=["mar","leo"],
                        palette_dyn_states ={"Erosional":"r","Recovery":"g","Depositional":"b","Vulnerability":"purple"},
                        orders=["Erosional","Recovery","Depositional","Vulnerability"],
                        xticks_labels=["Marengo","St. Leo."],
                           figsize=(8,4),
                           font_scale=0.8):


        f,ax=plt.subplots(figsize=figsize)

        sb.set_context("paper", font_scale=font_scale)
        sb.set_style("whitegrid")

        plot_bars=sb.barplot(data=self.location_ebcds,x="location", y=self.location_ebcds.coastal_markov_idx,hue="states_labels", hue_order=orders,
                  order=loc_order, palette=palette_dyn_states)


        ax.set_xticklabels(labels=xticks_labels)
        ax.set_xlabel("")
        ax.set_ylabel("e-BCD")


        txt_heights=[i.get_height() for i in ax.patches]

        signs=[]
        for i in orders:
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

    #f.savefig(r'E:\\path\\to\\save\\picture.png', dpi=600);


def LISA_site_level(
    dh_df,
    crs_dict_string,
    geometry_column="coordinates",
    mode="distance",
    distance_value=35,
    decay=-2,
    k_value=300,
):
    """Performs Hot-Spot analysis using Local Moran's I as LISA for all the survey.
        Please refer to PySAL package documentation for more info.

    Args:
        dh_df (df, str): Pandas dataframe or local path of the timeseries files,
        as returned by the multitemporal extraction.
        crs_dict_string (dict): Dictionary storing location codes as key and crs information as values, in dictionary form.
        Example: crs_dict_string = {'wbl': {'init': 'epsg:32754'},
                   'apo': {'init': 'epsg:32754'},
                   'prd': {'init': 'epsg:32755'},
                   'dem': {'init': 'epsg:32755'} }

        geometry_column (str): field storing the geometry column. If in string form (as loaded from a csv), it will be converted to Point objects. Default='coordinates'.

        mode (str)('distance','knn','idw'): If 'distance'(Default), compute spatial weigth matrix using a distance-band kernel, specified in distance_value parameter.
                                        If 'knn', spatial weigth matrix uses a specified (k_value parameter) of k number closest points to compute weigths.
                                        if 'idw', Inverse Distance Weigthing is used with the specified decay power (decay parameter) to compute weigth.

        distance_value (int): values in meters (crs must be projected) used as distance band for neigthours definition in distance weigth matrix computation.
        decay (int): power of decay to use with IDW.
        k_value (int): number of closest points for neigthours definition in distance weigth matrix computation.


    Returns:
        Dataframe with the fdr threshold, local moran-s Is, p and z values and the quadrant
        in which each observation falls in a Moran's scatter plot.
    """

    if isinstance(dh_df, str):
        if os.path.isfile(dh_df):
            df = pd.read_csv(dh_df)
        else:
            raise NameError ("The string provided in dh_df is not a valid path.")

    elif isinstance(dh_df, pd.DataFrame):
        df=dh_df

    else:
        raise NameError ("dh_df parameter must be either a valid path pointing to a multitemporal dataframe or the dataframe itself as a Pandas Dataframe.")

    lisa_df = pd.DataFrame()

    locs = df.location.unique()  # obtain list of locations

    # check wether a geometry type column is present
    if not isinstance((df.loc[0,geometry_column]), Point):
        df[geometry_column] = df[geometry_column].apply(wkt.loads)
    else:
        pass

    for loc in tqdm(locs):

        print(f"Working on {loc}")

        df_in = df.query(f"location=='{loc}'")  # subset a location

        # create a GeoDataFrame with the right CRS
        gdf = gpd.GeoDataFrame(df_in, geometry=geometry_column, crs=crs_dict_string[loc])

        dts = gdf.dt.unique()  # obtain list of periods

        for dt in tqdm(dts):

            gdf_input = gdf.query(f"dt=='{dt}'")  # subset a periods
            gdf_input.dropna(axis=0, how="any", subset=["dh"], inplace=True)
            # drop rows where dh is null, due to sand-only condition

            if mode == "distance":
                dist = distance_value
                dist_mode = "distance_band"
                decay = 0

                dist_w = weights.DistanceBand.from_dataframe(
                    df=gdf_input, threshold=dist, binary=True
                )
                # create a binary spatial weight matrix with no IDW and specified
                # distance
            if mode == "idw":
                dist = distance_value
                dist_mode = "idw"
                decay = decay

                dist_w = weights.DistanceBand.from_dataframe(
                    df=gdf_input, threshold=dist, binary=False, alpha=decay
                )
                # create a binary spatial weight matrix with no IDW and specified
                # distance

            elif mode == "knn":

                k = k_value
                dist_mode = "k"
                decay = 0
                dist = k_value

                dist_w = weights.distance.KNN.from_dataframe(df=gdf_input, k=int(k))

            lisa_ = moran.Moran_Local(gdf_input.dh, dist_w, permutations=999)

            fdr_lisa = fdr(lisa_.p_sim)  # as k
            # the False Discovery Rate threshold to use for significant cluster
            gdf_input["lisa_fdr"] = fdr_lisa
            # the quadrant of the Moran's scatter plot (Anselin 1995) in Pysal scheme
            gdf_input["lisa_q"] = lisa_.q
            gdf_input["lisa_I"] = lisa_.Is  # the local Moran's Is
            # the number of valid observations used
            gdf_input["lisa_n_val_obs"] = lisa_.n
            gdf_input["lisa_opt_dist"] = dist  # the distance used
            gdf_input["lisa_dist_mode"] = dist_mode  # mode, k od distance
            gdf_input["lisa_p_sim"] = lisa_.p_sim  # permutations (999) based p-value
            gdf_input["lisa_z_sim"] = lisa_.z_sim  # permutations (999) based z-value
            gdf_input["lisa_z"] = lisa_.z
            # z-value of the original data I distribution (no permutation)
            gdf_input["decay"] = decay

            lisa_df = pd.concat([lisa_df, gdf_input], ignore_index=True)

    return lisa_df
