"""Hotspot module."""

from tqdm.notebook import tqdm
import pandas as pd
from shapely.geometry import Point
from shapely import wkt
import geopandas as gpd
import numpy as np
from pysal.lib import weights
import pysal.explore.esda.moran as moran
from pysal.explore.esda.util import fdr
from sandpyper.outils import coords_to_points
from itertools import product as prod
from pysal.viz.mapclassify import (EqualInterval,
                                   FisherJenks,
                                   HeadTailBreaks,
                                   JenksCaspall,
                                   KClassifiers,
                                   Quantiles,
                                   Percentiles,
                                   UserDefined)


class Discretiser:
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

    def __init__(self, bins, method, labels=None):

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



    def fit(self, df, absolute = True, field="dh", appendix = ("_deposition","_erosion"), print_summary=False):
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

        if absolute:
            data_ero=df[df.loc[:,field] < 0]
            data_depo=df[df.loc[:,field] > 0]
            data=np.abs(df.loc[:,field])
        else:
            data=df.loc[:,field]

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

        states_depo={yb:class_name for yb,class_name in enumerate(depo_class_names)}
        states_ero={yb*-1:class_name for yb,class_name in enumerate(ero_class_names, start=-len(ero_class_names)+1)}

        data_ero["markov_tag"]=[states_ero[i] for i in class_erosion]
        data_depo["markov_tag"]=[states_depo[i] for i in class_deposition]


        return pd.concat([data_ero,data_depo],ignore_index=False)


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
