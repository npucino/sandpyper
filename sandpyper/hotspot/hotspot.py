import numpy as np
from tqdm import tqdm_notebook as tqdm
import pandas as pd
import geopandas as gpd

from pysal.lib import weights
import pysal.explore.esda.moran as moran
from pysal.explore.esda.util import fdr



def LISA_site_level (dh_path,crs_dict_string,unique_field="coordinates",
                    mode='distance',distance_value=35,decay=-2,k_value=300):
    """ Performs Hot-Spot analysis using Local Moran's I as LISA for all the survey.
        Please refer to PySAL package documentation for more info.

    Args:
        dh_path (str): Local path of the timeseries files,
        as returned by the multitemporal extraction.

        crs_dict_string (dict): Dictionary storing location codes as key and crs information as values, in dictionary form.
        Example: crs_dict_string = {'wbl': {'init': 'epsg:32754'},
                   'apo': {'init': 'epsg:32754'},
                   'prd': {'init': 'epsg:32755'},
                   'dem': {'init': 'epsg:32755'} }

        unique_field (str): field storing unique Spatial IDs, relative to space and not time. Default='coordinates'.

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
    df=pd.read_csv(dh_path)

    lisa_df=pd.DataFrame()

    locs=df.location.unique()  # obtain list of locations

    for loc in tqdm(locs):

        print(f"Working on {loc}")

        df_in=df.query(f"location=='{loc}'") # subset a location
        df_in['geometry']=df_in.loc[:,unique_field].apply(coords_to_points)   # recreate geometry field
        gdf = gpd.GeoDataFrame(df_in, geometry='geometry', crs=crs_dict_string[loc]) # create a GeoDataFrame with the right CRS

        dts=gdf.dt.unique() # obtain list of periods

        for dt in tqdm(dts):

            gdf_input=gdf.query(f"dt=='{dt}'") # subset a periods
            gdf_input.dropna(axis=0, how='any', subset=['dh'], inplace=True)
            # drop rows where dh is null, due to sand-only condition

            if mode == "distance":
                dist=distance_value
                optimal_distance=dist        # USELESS
                dist_mode="distance_band"
                decay=0

                dist_w=weights.DistanceBand.from_dataframe(df=gdf_input,threshold=dist,binary=True)
                # create a binary spatial weight matrix with no IDW and specified distance
            if mode == "idw":
                dist=distance_value
                optimal_distance=dist        # USELESS
                dist_mode="idw"
                decay=decay

                dist_w=weights.DistanceBand.from_dataframe(df=gdf_input,threshold=dist,binary=False, alpha=decay)
                # create a binary spatial weight matrix with no IDW and specified distance

            elif mode =="knn":

                k=k_value
                optimal_distance=k
                dist_mode="k"
                decay=0

                # TO DO:
                # due to the pooling strategy, if k is bigger than the total number of valid observations, then
                # try to use the survey k mean. If also the mean survey k is bigger, than use half of the valid observations
                # as k.

                dist_w=weights.distance.KNN.from_dataframe(df=gdf_input,k=int(k))



            lisa_ = moran.Moran_Local(gdf_input.dh,dist_w, permutations=999)

            fdr_lisa=fdr(lisa_.p_sim)     # as k
            gdf_input["lisa_fdr"]=fdr_lisa   # the False Discovery Rate threshold to use for significant cluster
            gdf_input["lisa_q"]=lisa_.q      # the quadrant of the Moran's scatter plot (Anselin 1995) in Pysal scheme
            gdf_input["lisa_I"]=lisa_.Is     # the local Moran's Is
            gdf_input["lisa_n_val_obs"]=lisa_.n        # the number of valid observations used
            gdf_input["lisa_opt_dist"]=dist      # the distance used
            gdf_input["lisa_dist_mode"]=dist_mode # mode, k od distance
            gdf_input["lisa_p_sim"]=lisa_.p_sim # permutations (999) based p-value
            gdf_input["lisa_z_sim"]=lisa_.z_sim # permutations (999) based z-value
            gdf_input["lisa_z"]=lisa_.z
            gdf_input["decay"]=decay        #  z-value of the original data I distribution (no permutation)


            lisa_df=pd.concat([lisa_df,gdf_input], ignore_index=True)

        return lisa_df
