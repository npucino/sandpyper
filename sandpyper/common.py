"""This module contains some common functions for both ProfileSet and ProfileDynamics classes and the upcoming Space module.
"""

import os
import re
import random
import glob
import datetime
import time
import warnings
import pickle
from operator import itemgetter
from shutil import move, copy

import numpy as np
import pandas as pd
import geopandas as gpd
import seaborn as sb
from tqdm.notebook import tqdm_notebook as tqdm
from fuzzywuzzy import fuzz
import itertools
from itertools import product, groupby

import math
from math import tan, radians, sqrt


import shapely
from shapely import wkt
from shapely.ops import split, snap
from shapely.geometry import Point, Polygon, LineString, mapping, box
from shapely.ops import split, snap, unary_union


from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score, mean_squared_error

import rasterio as ras
from rasterio import features
import rasterio.mask as rasmask
from rasterio.windows import Window
from rasterio.transform import rowcol
from rasterio.io import MemoryFile


from skimage.filters import threshold_multiotsu

from scipy.stats import median_abs_deviation, shapiro, normaltest
from scipy.ndimage import gaussian_filter
import scipy.signal as sig
from scipy.integrate import simps


import richdem as rd

from statsmodels.api import qqplot

from pysal.lib import weights
import pysal.explore.esda.moran as moran
from pysal.explore.esda.util import fdr

from pysal.explore.giddy.markov import Markov
from pysal.lib.weights import DistanceBand, Queen, higher_order
from pysal.viz.mapclassify import (EqualInterval,
                                   FisherJenks,
                                   HeadTailBreaks,
                                   JenksCaspall,
                                   KClassifiers,
                                   Quantiles,
                                   Percentiles,
                                   UserDefined)



import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import dates
from matplotlib.dates import DateFormatter
from matplotlib.ticker import FixedFormatter, FixedLocator, AutoMinorLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


######################################


from rasterio.windows import Window
from rasterio.transform import rowcol
import rasterio as ras
import numpy as np
import richdem as rd
from shapely.geometry import Point
import pandas as pd
import geopandas as gpd
from tqdm.notebook import tqdm

import os
import time
import warnings
import pickle


# TO MODIFY IN SPACE FUNCTIONS #################################
import itertools as it



######### TOOLS FUNCTIONS ###################

def test_format(filename, loc_search_dict):
    """
    It returns True if the filename matches the required format (regx) or False if it doesn't.

    Args:
        filename (str): filename to test, of the type "Seaspray_22_Oct_2020_GeoTIFF_DSM_GDA94_MGA_zone_55.tiff".
        loc_search_dict (dict): a dictionary where keys are the location codes and values are lists containing the expected full location string (["Warrnambool", "warrnambool","warrny"]).
    Returns:
        bool (bool): True if the filename matches the required format (regx) or False if it doesn't.
    """

    re_list_loc = "|".join(loc_search_dict.keys())
    regx = rf"\d{{8}}_({re_list_loc})_(ortho|dsm)\.(tiff|tif)"

    try:
        re.search(regx, filename).group()
        return True
    except:
        return False


def find_date_string(
    filename,
    list_months=[
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sept",
        "oct",
        "nov",
        "dec",
    ],
    to_rawdate=True,
):
    """It finds the date in the filename and returns True if it is already in the Sandpyper format. If it is not formatted and a date is found, it returns a formatted version of it. For example, dates in raw filenames should be similar to "Seaspray_22_Oct_2020_GeoTIFF_DSM_GDA94_MGA_zone_55.tiff".

    Args:
        filename (str): filename to test, of the type "Seaspray_22_Oct_2020_GeoTIFF_DSM_GDA94_MGA_zone_55.tiff".
        list_months (list): expected denominations for the months. Default to ['jan','feb','mar',...,'dec'].
        to_rawdate (bool): True to format the found date into raw_date (20201022). False, return True if the date is found or False if not.

    Returns:
        bool (bool, str): True if it was already formatted or a new string formatted correctly.
    """

    re_list_months = "|".join(list_months)
    regx = rf"_\d{{2}}_({re_list_months})_\d{{4}}_"

    try:
        group = re.search(regx, filename, re.IGNORECASE).group()
        if to_rawdate == False:
            return True
        else:
            group_split = group.split("_")
            dt = datetime.datetime.strptime(
                f"{group_split[1]}{group_split[2]}{group_split[3]}", "%d%b%Y"
            )
            return dt.strftime("%Y%m%d")
    except:
        return False


def filter_filename_list(filenames_list, fmt=[".tif", ".tiff"]):
    """It returns a list of only specific file formats from a list of filenames.

    Args:
        filenames_list (list): list of filenames.
        fmt (list): list of formats to be filtered (DEFAULT = [".tif",".tiff"])

    Returns:
        filtered_list (list): A filtered list of filenames.
    """
    return [name for name in filenames_list if os.path.splitext(name)[1] in fmt]


def round_special(num, thr):
    """It rounds the number (num) to its closest fraction of threshold (thr). Useful to space ticks in plots."""
    return round(float(num) / thr) * thr


def coords_to_points(string_of_coords):
    """
    Function to create Shapely Point geometries from strings representing Shapely Point geometries.
    Used when loading CSV with point geometries in string type.

    args:
        string_of_coords (str): the string version of Shapely Point geometry

    returns:
        pt_geom : Shapely Point geometry
    """
    num_ditis = re.findall("\\d+", string_of_coords)
    try:
        coord_x = float(num_ditis[0] + "." + num_ditis[1])
        coord_y = float(num_ditis[2] + "." + num_ditis[3])
        pt_geom = Point(coord_x, coord_y)
    except BaseException:
        print(
            f"point creation failed! Assigning NaN. Check the format of the input string."
        )
        pt_geom = np.nan
    return pt_geom


def create_id(
    series,
    tr_id_field="tr_id",
    loc_field="location",
    dist_field="distance",
    random_state=42,
):
    """Function to create unique IDs from random permutations of integers and letters from the distance, tr_id, location, coordinates and survey_date fields of the rgb and z tables.

    args:
        series (pd.Series): series having the selected fields.
        tr_id_field (str): Field name holding the transect ID (Default="tr_id").
        loc_field (str): Field name holding the location of the survey (Default="location").
        dist_field (str): Field name holding the distance from start of the transect (Default="distance").
        random_state (int): Random seed.

    returns:
        ids (list): A series od unique IDs.
    """

    dist_c = str(np.round(float(series.loc[dist_field]), 2))
    tr_id_c = str(series.loc[tr_id_field])
    loc_d = str(series.loc[loc_field])

    if type(series.coordinates) != str:
        coord_c = series.coordinates.wkt.split()[1][-3:]
    else:
        coord_c = str(series.coordinates.split()[1][-3:])

    if type(series.survey_date) != str:
        date_c = str(series.survey_date.date())
    else:
        date_c = str(series.survey_date)

    ids_tmp = dist_c + "0" + tr_id_c + loc_d + coord_c + date_c

    ids = ids_tmp.replace(".", "0").replace("-", "")
    char_list = list(ids)  # convert string inti list
    random.Random(random_state).shuffle(
        char_list,
    )  # shuffle the list
    ids = "".join(char_list)

    return ids


def create_spatial_id(series, random_state=42):
    """Function to create IDs indipended on the survey_date, but related to to distance, tr_id and location only. Equivalent to use coordinates field.

    args:
        series (pd.Series): series of merged table.
        random_state (int): Random seed.

    returns:
        ids (list): A series od unique spatial IDs.
    """

    # ID indipended on the survey_date, but only related to distance, tr_id
    # and location. Useful ID, but equivalent to use coordinates field.

    ids = (
        str(np.round(float(series.distance), 2))
        + "0"
        + str(series.tr_id)
        + str(series.location)
    )
    ids = ids.replace(".", "0").replace("-", "")
    char_list = list(ids)  # convert string inti list
    random.Random(random_state).shuffle(
        char_list,
    )  # shuffle the list
    ids = "".join(char_list)

    return ids


def getListOfFiles(dirName):
    """
    Function to create a list of files from a folder path, including sub folders.

    Args:
        dirName (str): Path of the parent directory.

    Returns:
        allFiles : list of full paths of all files found.
    """

    # create a list of file and sub directories names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()  # Iterate over all the entries
    for entry in listOfFile:

        fullPath = os.path.join(dirName, entry)  # Create full path

        if os.path.isdir(
            fullPath
        ):  # If entry is a directory then get the list of files in this directory
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles


def getLoc(filename, list_loc_codes):
    """
    Function that returns the location code from properly formatted (see documentation) filenames.

    Args:
        filename (str): filename (i.e. apo_20180912_dsm_ahd.tiff).
        list_loc_codes (list): list of strings containing location codes.

    Returns:
        str : location codes.
    """

    return next((x for x in list_loc_codes if x in filename), False)


def getDate(filename):
    """
    Returns the date in raw form (i.e 20180912) from already formatted filenames.

    Args:
        filename (str): filename (i.e. apo_20180912_dsm_ahd.tiff).

    Returns:
        str : raw date.
    """
    # get the date out of a file input

    num_ditis = re.findall("\\d+", filename)

    # now we need to convert each number into integer. int(string) converts string into integer
    # we will map int() function onto all elements of numbers list
    num_ditis = map(int, num_ditis)
    try:
        date_ = max(num_ditis)
        if len(str(date_)) == 8:
            return date_
        else:
            print(f"Unable to find correct date from input filename. Found: {date_}.")
    except BaseException:
        raise TypeError(print("No numbers in the input filename."))

    return max(num_ditis)


def getListOfDate(list_dsm):
    """
    Returns the a list of raw dates (i.e 20180912) from a list of formatted filenames.

    Args:
        list_dsm (list): list of filenames of DSM or rothophotos datasets.

    Returns:
        list : raw dates.
    """
    dates = []
    for i in list_dsm:
        temp = getDate(i)
        dates.append(temp)
    return dates


def extract_loc_date(name, loc_search_dict, split_by="_"):

    """

    Get the location code (e.g. wbl, por) and raw dates (e.g. 20180902) from filenames using the search dictionary.
    If no location is found using exact matches, a fuzzy word match is implemented, searching closest matches
    between locations in filenames and search candidates provided in the loc_search_dict dictionary.

    Args:

        name (str): full filenames of the tipy 'C:\\jupyter\\data_in_gcp\\20180601_mar_gcps.csv').
        loc_search_dict (dict): a dictionary where keys are the location codes and values are lists containing the expected full location string (["Warrnambool", "warrnambool","warrny"]).
        split_by (str): the character used to split the name (default= '_').

    Returns:

        ('location',raw_date) : tuple with location and raw date.

    """

    try:

        date=getDate(name)

    except:

        print("Proceeding with automated regular expression match")

        date=find_date_string(name)

        print(f"Date found: {date}")



    names = set((os.path.split(name)[-1].split(split_by)))

    locations_search_names=list(loc_search_dict.values())
    locations_codes=list(loc_search_dict.keys())

    for loc_code, raw_strings_loc in zip(locations_codes, locations_search_names):  # loop trhough all possible lists of raw strings

        raw_str_set = set(raw_strings_loc)

        match = raw_str_set.intersection(names)

        if len(match) >= 1:

            location_code_found = loc_code

            break

    try:
        return (location_code_found, date)

    except:
        # location not found. Let's implement fuzzy string match.

        scores =[]
        for i,search_loc in enumerate(locations_search_names):
            for word in search_loc:
                score=fuzz.ratio(word,names) # how close is each candidate word to the list of names which contain the location?
                scores.append([score,i,word])

        scores_arr=np.array(scores) # just to safely use np.argmax on a specified dimension

        max_score_idx=scores_arr[:,:2].astype(int).argmax(0)[0] # returns the index of the maximum score in scores array
        closest_loc_code_idx=scores[max_score_idx][1] # closest code found

        closest_word=scores[max_score_idx][-1]
        loc_code=locations_codes[closest_loc_code_idx]

        print(f"Location not understood in {name}.\n\
        Fuzzy word matching found {closest_word}, which corresponds to location code: {loc_code} ")

        return (loc_code, date)

def getCrs_from_raster_path(ras_path):
    """Returns the EPSG code of the input raster (geotiff).

    Args:
        ras_path (str): Path of the raster.

    Returns:
        epsg_code (int): EPSG code of the input raster.
    """
    with ras.open(r"{}".format(ras_path)) as raster:
        return raster.crs.to_epsg()

def filepath_raster_type(raster_path):
        """Returns 'dsm' if the path provided (raster_path) is of a DSM (1 single band) or 'ortho' if has multiple bands."""
        with ras.open(r"{}".format(raster_path)) as raster:
            if raster.count==1:
                return "dsm"
            else:
                return "ortho"

def getCrs_from_transect(trs_path):
    """Returns the EPSG code of the input transect file (geopackage).

    Args:
        trs_path (str): Path of the transect file.

    Returns:
        epsg_code (int): EPSG code of the input transect file.
    """
    return gpd.read_file(trs_path).crs

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

def cross_ref(
    dir_inputs, dirNameTrans, loc_search_dict, list_loc_codes, print_info=False
):
    """Returns a dataframe with location, raw_date, filenames (paths) and CRS of each raster and associated transect files. Used to double-check.

    args:
        dir_inputs (str): Path of the directory containing the geotiffs datasets (.tiff or .tif).
        dirNameTrans (str): Path of the directory containing the transects (geopackages, .gpkg).
        loc_search_dict (list): Dictionary used to match filename with right location code.
        list_loc_codes (list): list of strings containing location codes.
        print_info (bool): If True, prints count of datasets/location and total. Default = False.

    returns:
        rasters_df (pd.DataFRame): Dataframe and information about raster-transect files matches.
    """

    ras_type_dict={0:"dsm",
                  1:"ortho"}

    # DF transects
    list_transects = glob.glob(rf"{dirNameTrans}/*.gpkg")
    locs_transects = pd.DataFrame(
        pd.Series(
            [getLoc(trs, list_loc_codes) for trs in list_transects], name="location"
        )
    )

    df_tmp_trd = pd.DataFrame(locs_transects, columns=["location"])
    df_tmp_trd["filename_trs"] = list_transects
    df_tmp_trd["crs_transect"] = df_tmp_trd.filename_trs.apply(getCrs_from_transect)

    if isinstance(dir_inputs, list):
        dirs=dir_inputs
        list_rasters_check_dsm=[]
        list_rasters_check_orthos=[]
    else:
        dirs=[dir_inputs]

    rasters_df=pd.DataFrame()

    for i,path_in in enumerate(dirs):

        list_rasters = glob.glob(rf"{path_in}/*.ti*")
        raster_types=[filepath_raster_type(i) for i in list_rasters]

        print(path_in)

        if len(set(raster_types)) != 1:
            raise ValueError(f"Mixed input types have been found in {ras_type_dict[i]} folder. Each folder has to contain either DSMs or orthos only.")

        if isinstance(dir_inputs, list):
            if i == 0:
                list_rasters_check_dsm.append([extract_loc_date(os.path.split(i)[-1],loc_search_dict) for i in list_rasters])
            else:
                list_rasters_check_orthos.append([extract_loc_date(os.path.split(i)[-1],loc_search_dict) for i in list_rasters])

        loc_date_labels_raster = [
            extract_loc_date(file1, loc_search_dict=loc_search_dict)
            for file1 in list_rasters
        ]

        df_tmp_raster = pd.DataFrame(
            loc_date_labels_raster, columns=["location", "raw_date"]
        )
        df_tmp_raster["filename_raster"] = list_rasters
        df_tmp_raster["crs_raster"] = df_tmp_raster.filename_raster.apply(
            getCrs_from_raster_path
        )
        df_tmp_raster["raster_type"]=raster_types

        rasters_df=pd.concat([rasters_df,df_tmp_raster], ignore_index=True)


    if isinstance(dir_inputs, list):

            missing_dsms=set(*list_rasters_check_orthos).difference(set(*list_rasters_check_dsm))
            missing_orthos=set(*list_rasters_check_dsm).difference(set(*list_rasters_check_orthos))

            if len(missing_dsms) >0:
                print(f"WARNING: {missing_dsms} missing or misnamed from the DSMs folder.\n")
            else:
                pass

            if len(missing_orthos)>0:
                print(f"WARNING: {missing_orthos} missing or misnamed from orthos folder.\n")
            else:
                pass

    matched = pd.merge(rasters_df, df_tmp_trd, on="location", how="left").set_index(
        ["location"]
    )
    check_formatted=pd.merge(matched.query("raster_type=='ortho'").reset_index(),
         matched.query("raster_type=='dsm'").reset_index(),
        on=["location","raw_date"],suffixes=("_ortho","_dsm"))

    check_formatted=check_formatted.loc[:,[ 'raw_date','location','filename_trs_ortho','crs_transect_dsm',
     'filename_raster_dsm','filename_raster_ortho',
    'crs_raster_dsm','crs_raster_ortho']]


    if bool(print_info) is True:
        counts=matched.groupby(["location","raster_type"]).count().reset_index()

        for i,row in counts.iterrows():
            print(f"{row['raster_type']} from {row['location']} = {row['raw_date']}\n")

        print(f"\number OF DATASETS TO PROCESS: {counts.raw_date.sum()}")

    return check_formatted

def spatial_id(geometry):

    if isinstance(geometry, str):
        geom_str=list(geometry.replace(" ",""))
    else:
        geom=geometry.to_wkt()
        geom_str=list(geom.replace(" ",""))

    geom_str_rnd=random.Random(42).shuffle(geom_str)
    return "".join(geom_str)

def create_details_df (dh_df, loc_full, fmt='%Y%m%d'):


    locs_dt_str=pd.DataFrame()
    for location in dh_df.location.unique():

        df_time_tmp=dh_df.query(f"location=='{location}'").groupby(['dt'])[['date_pre','date_post']].first().reset_index()
        df_time_tmp["orderid"]=[int(i.split("_")[1]) for i in df_time_tmp.dt]
        df_time_tmp.sort_values(["orderid"], inplace=True)
        df_time_tmp["location"]=location
        locs_dt_str=pd.concat([df_time_tmp,locs_dt_str], ignore_index=True)

    # add days between dates
    deltas=[(datetime.datetime.strptime(str(d_to), fmt) - datetime.datetime.strptime(str(d_from), fmt)).days
            for d_to,d_from in zip(locs_dt_str.date_post,locs_dt_str.date_pre)]
    locs_dt_str['n_days']=deltas

    # add full names to codes
    locs_dt_str['loc_full']=locs_dt_str.location.map(loc_full)

    # some cleaning and renaming
    locs_dt_str.drop('orderid',1,inplace=True)

    return locs_dt_str



def getAngle(pt1, pt2):
    """Helper function to return the angle of two points (pt1 and pt2) coordinates in degrees.
    Source: http://wikicode.wikidot.com/get-angle-of-line-between-two-points"""

    x_diff = pt2[0] - pt1[0]
    y_diff = pt2[1] - pt1[1]
    return math.degrees(math.atan2(y_diff, x_diff))


def getPoint1(pt, bearing, dist):
    """Helper function to return the point coordinates at a determined distance (dist) and bearing from a starting point (pt)."""

    angle = bearing + 90
    bearing = math.radians(angle)
    x = pt[0] + dist * math.cos(bearing)
    y = pt[1] + dist * math.sin(bearing)
    return Point(x, y)


def getPoint2(pt, bearing, dist):
    bearing = math.radians(bearing)
    x = pt[0] + dist * math.cos(bearing)
    y = pt[1] + dist * math.sin(bearing)
    return Point(x, y)


def split_transects(geom, side="left"):
    """Helper function to split a transect geometry along its centroid, retaining only their left (default) or right side.

    args:
        geom (geometry): geometry (shapely LineString) of the transect to split.
        side (str): side to keep ('left' or 'right').

    returns:
        geometry: New geometry split.
    """
    side_dict = {"left": 0, "right": 1}
    snapped = snap(geom, geom.centroid, 0.001)
    result = split(snapped, geom.centroid)
    return result[side_dict[side]]


def create_transects(baseline, sampling_step, tick_length, location, crs, side="both"):
    """Creates a GeoDataFrame with transects normal to the baseline, with defined spacing and length.

    args:
        baseline (gdp.GeoDataFrame): baseline geodataframe.
        sampling_step (int,float): alognshore spacing of transects in the CRS reference unit.
        tick_length (int,float): transects length
        location (str): location code
        crs (dict): coordinate reference system to georeference the transects. It must be in dictionary form.
        side (str): If "both", the transects will be centered on the baseline. If "left" or "right", transects will start from the baseline and extend to the left/right of it.

    returns:
        gdf_transects (gpd.GeoDataFrame): Geodataframe of transects.
    """

    if side != "both":
        tick_length = 2 * tick_length
    else:
        pass

    if sampling_step == 0 or sampling_step >= baseline.length.values[0]:
        raise ValueError(f"Sampling step provided ({sampling_step}) cannot be zero or equal or greater than the baseline length ({baseline.length.values[0]}).")
    else:
        try:
            dists = np.arange(0, baseline.geometry.length[0], sampling_step)
        except BaseException:
            try:
                dists = np.arange(0, baseline.geometry.length, sampling_step)
            except BaseException:
                dists = np.arange(0, baseline.geometry.length.values[0], sampling_step)

        points_coords = []
        try:
            for j in [baseline.geometry.interpolate(i) for i in dists]:
                points_coords.append((j.geometry.x[0], j.geometry.y[0]))
        except BaseException:
            for j in [baseline.geometry.interpolate(i) for i in dists]:
                points_coords.append((j.geometry.x, j.geometry.y))

                # create transects as Shapely linestrings

        ticks = []
        for num, pt in enumerate(points_coords, 1):
            # start chainage 0
            if num == 1:
                angle = getAngle(pt, points_coords[num])
                line_end_1 = getPoint1(pt, angle, tick_length / 2)
                angle = getAngle([line_end_1.x, line_end_1.y], pt)
                line_end_2 = getPoint2([line_end_1.x, line_end_1.y], angle, tick_length)
                tick = LineString(
                    [(line_end_1.x, line_end_1.y), (line_end_2.x, line_end_2.y)]
                )

            ## everything in between
            if num < len(points_coords) - 1:
                angle = getAngle(pt, points_coords[num])
                line_end_1 = getPoint1(points_coords[num], angle, tick_length / 2)
                angle = getAngle([line_end_1.x, line_end_1.y], points_coords[num])
                line_end_2 = getPoint2([line_end_1.x, line_end_1.y], angle, tick_length)
                tick = LineString(
                    [(line_end_1.x, line_end_1.y), (line_end_2.x, line_end_2.y)]
                )

            # end chainage
            if num == len(points_coords):
                angle = getAngle(points_coords[num - 2], pt)
                line_end_1 = getPoint1(pt, angle, tick_length / 2)
                angle = getAngle([line_end_1.x, line_end_1.y], pt)
                line_end_2 = getPoint2([line_end_1.x, line_end_1.y], angle, tick_length)
                tick = LineString(
                    [(line_end_1.x, line_end_1.y), (line_end_2.x, line_end_2.y)]
                )

            ticks.append(tick)

        gdf_transects = gpd.GeoDataFrame(
            {
                "tr_id": range(len(ticks)),
                "geometry": ticks,
                "location": [location for i in range(len(ticks))],
            },
            crs=crs,
        )

        # clip the transects

        if side == "both":
            pass
        else:

            gdf_transects["geometry"] = gdf_transects.geometry.apply(
                split_transects, **{"side": side}
            )

        return gdf_transects






######### PROFILE FUNCTIONS ###################

def get_terrain_info(x_coord, y_coord, rdarray):
    """
    Returns the value of the rdarray rasters.

    Args:
        x_coord (float): Projected X coordinates of pixel to extract value.
        y_coord (float): Projected Y coordinates of pixel to extract value.
        rdarray (rdarray): rdarray dataset.

    Returns:
        px (int, float): rdarray pixel value.
    """

    geotransform = rdarray.geotransform

    xOrigin = geotransform[0]  # top-left X
    yOrigin = geotransform[3]  # top-left y
    pixelWidth = geotransform[1]  # horizontal pixel resolution
    pixelHeight = geotransform[5]  # vertical pixel resolution
    px = int((x_coord - xOrigin) / pixelWidth)  # transform geographic to image coords
    py = int((y_coord - yOrigin) / pixelHeight)  # transform geographic to image coords

    try:
        return rdarray[py, px]
    except BaseException:
        return np.nan


def get_elevation(x_coord, y_coord, raster, bands, transform):
    """
    Returns the value of the raster at a specified location and band.

    Args:
        x_coord (float): Projected X coordinates of pixel to extract value.
        y_coord (float): Projected Y coordinates of pixel to extract value.
        raster (rasterio open file): Open raster object, from rasterio.open(raster_filepath).
        bands (int): number of bands.
        transform (Shapely Affine obj): Geotransform of the raster.
    Returns:
        px (int,float): raster pixel value.
    """
    elevation = []
    row, col = rowcol(transform, x_coord, y_coord, round)

    for j in np.arange(bands):  # we could iterate thru multiple bands

        try:
            data_z = raster.read(1, window=Window(col, row, 1, 1))
            elevation.append(data_z[0][0])
        except BaseException:
            elevation.append(np.nan)

    return elevation


def get_raster_px(x_coord, y_coord, raster, bands=None, transform=None):

    if isinstance(raster, richdem.rdarray):
        transform = rdarray.geotransform

        xOrigin = transform[0]  # top-left X
        yOrigin = transform[3]  # top-left y
        pixelWidth = transform[1]  # horizontal pixel resolution
        pixelHeight = transform[5]  # vertical pixel resolution
        px = int(
            (x_coord - xOrigin) / pixelWidth
        )  # transform geographic to image coords
        py = int(
            (y_coord - yOrigin) / pixelHeight
        )  # transform geographic to image coords

        try:
            return rdarray[py, px]
        except BaseException:
            return np.nan

    else:
        if bands == None:
            bands = raster.count()

        if bands == 1:
            try:
                px_data = raster.read(1, window=Window(col, row, 1, 1))
                return px_data[0][0]
            except BaseException:
                return np.nan
        elif bands > 1:
            px_data = []
            for band in range(1, bands + 1):
                try:
                    px_data_band = raster.read(band, window=Window(col, row, 1, 1))
                    px_data.append(px_data_band[0][0])
                except BaseException:
                    px_data.append(np.nan)

            return px_data


def get_profiles(
    dsm,
    transect_file,
    tr_ids,
    transect_index,
    step,
    location,
    date_string,
    add_xy=False,
    add_terrain=False,
):
    """
    Returns a tidy GeoDataFrame of profile data, extracting raster information
    at a user-defined (step) meters gap along each transect.

    Args:
        dsm (str): path to the DSM raster.
        transect_file (str): path to the transect file.
        transect_index (int): index of the transect to extract information from.
        step (int,float): sampling distance from one point to another in meters along the transect.
        location (str): location code
        date_string (str): raw format of the survey date (20180329)
        add_xy (bool): True to add X and Y coordinates fields.
        add_terrain (bool): True to add slope in degrees. Default to False.

    Returns:
        gdf (gpd.GeoDataFrame) : Profile data extracted from the raster.
    """

    ds = ras.open(dsm, "r")
    bands = ds.count  # get raster bands. One, in a classic DEM
    transform = ds.transform  # get geotransform info

    # index each transect and store it a "line" object
    line = transect_file.loc[transect_index]

    if tr_ids=='reset':
        line_id=line.name
    elif isinstance(tr_ids,str) and tr_ids in line.index:
        line_id=line.loc[tr_ids]
    else:
        raise ValueError(f"'tr_ids' must be either 'reset' or the name of an existing column o the transect files. '{tr_ids}' was passed.")


    length_m = line.geometry.length

    # Creating empty lists of coordinates, elevations and distance (from start
    # to end points along each transect lines)

    x = []
    y = []
    z = []
    slopes = []

    # The "distance" object is and empty list which will contain the x variable
    # which is the displacement from the shoreward end of the transects toward
    # the foredunes.

    distance = []

    for currentdistance in np.arange(0, int(length_m), step):

        # creation of the point on the line
        point = line.geometry.interpolate(currentdistance)
        xp, yp = (
            point.x,
            point.y,
        )  # storing point xy coordinates into xp,xy objects, respectively
        x.append(xp)  # see below
        y.append(
            yp
        )  # append point coordinates to previously created and empty x,y lists
        # extraction of the elevation value from DSM
        z.append(get_elevation(xp, yp, ds, bands, transform)[0])
        if str(type(add_terrain)) == "<class 'richdem.rdarray'>":
            slopes.append(get_terrain_info(xp, yp, add_terrain))
        else:
            pass

        # append the distance value (currentdistance) to distance list
        distance.append(currentdistance)

    # Below, the empty lists tr_id_list and the date_list will be filled by strings
    # containing the transect_id of every point as stored in the original dataset
    # and a label with the date as set in the data setting section, after the input.

    zs= pd.Series((elev for elev in z))

    if str(type(add_terrain)) == "<class 'richdem.rdarray'>":
        slopes_series= pd.Series((slope for slope in slope))
        df = pd.DataFrame({"distance": distance, "z": zs, "slope":slopes_series})
    else:
        df = pd.DataFrame({"distance": distance, "z": zs})


    df["coordinates"] = list(zip(x, y))
    df["coordinates"] = df["coordinates"].apply(Point)
    df["location"] = location
    df["survey_date"] = pd.to_datetime(date_string, yearfirst=True, dayfirst=False, format="%Y%m%d")
    df["raw_date"] = date_string
    df["tr_id"] = int(line_id)
    gdf = gpd.GeoDataFrame(df, geometry="coordinates")


    # The proj4 info (coordinate reference system) is gathered with
    # Geopandas and applied to the newly created one.
    gdf.crs = str(transect_file.crs)

    # Transforming non-hashable Shapely coordinates to hashable strings and
    # store them into a variable

    # Let's create unique IDs from the coordinates values, so that the Ids
    # follows the coordinates
    gdf["point_id"] = [create_id(gdf.iloc[i]) for i in range(0, gdf.shape[0])]

    if bool(add_xy):
        # Adding long/lat fields
        gdf["x"] = gdf.coordinates.x
        gdf["y"] = gdf.coordinates.y
    else:
        pass

    return gdf


def get_dn(x_coord, y_coord, raster, bands, transform):
    """Returns the value of the raster at a specified location and band.

    args:
        x_coord (float): Projected X coordinates of pixel to extract value.
        y_coord (float): Projected Y coordinates of pixel to extract value.
        raster (rasterio open file): Open raster object, from rasterio.open(raster_filepath).
        bands (int): number of bands.
        transform (Shapely Affine obj): Geotransform of the raster.

    returns:
        px (int, float): raster pixel value.
    """
    # Let's create an empty list where we will store the elevation (z) from points
    # With GDAL, we extract 4 components of the geotransform (gt) of our north-up image.

    dn_val = []
    row, col = rowcol(transform, x_coord, y_coord, round)

    for j in range(1, 4):  # we could iterate thru multiple bands

        try:
            data = raster.read(j, window=Window(col, row, 1, 1))
            dn_val.append(data[0][0])
        except BaseException:
            dn_val.append(np.nan)
    return dn_val


def get_profile_dn(
    ortho, transect_file,
    transect_index, tr_ids,
    step, location, date_string, add_xy=False
):
    """Returns a tidy GeoDataFrame of profile data, extracting raster information at a user-defined (step) meters gap along each transect.

    Args:
        ortho (str): path to the DSM raster.
        transect_file (str): path to the transect file.
        transect_index (int): index of the transect to extract information from.
        step (int,float): sampling distance from one point to another in meters along the transect.
        location (str): location code
        date_string (str): raw format of the survey date (20180329)
        add_xy (bool): True to add X and Y coordinates fields.

    Returns:
        gdf (gpd.GeoDataFrame) : Profile data extracted from the raster.
    """

    ds = ras.open(ortho, "r")

    bands = ds.count

    transform = ds.transform

    line = transect_file.loc[transect_index]

    if tr_ids=='reset':
        line_id=line.name
    elif isinstance(tr_ids,str) and tr_ids in line.index:
        line_id=line.loc[tr_ids]
    else:
        raise ValueError(f"'tr_ids' must be either 'reset' or the name of an existing column o the transect files. '{tr_ids}' was passed.")


    length_m = line.geometry.length

    x = []
    y = []
    dn = []
    distance = []
    for currentdistance in np.arange(0, int(length_m), step):
        # creation of the point on the line
        point = line.geometry.interpolate(currentdistance)
        xp, yp = (
            point.x,
            point.y,
        )  # storing point xy coordinates into xp,xy objects, respectively
        x.append(xp)  # see below
        y.append(
            yp
        )  # append point coordinates to previously created and empty x,y lists
        dn.append(get_dn(xp, yp, ds, bands, transform))

        distance.append(currentdistance)

    dn1 = pd.Series((v[0] for v in dn))
    dn2 = pd.Series((v[1] for v in dn))
    dn3 = pd.Series((v[2] for v in dn))
    df = pd.DataFrame({"distance": distance, "band1": dn1, "band2": dn2, "band3": dn3})
    df["coordinates"] = list(zip(x, y))
    df["coordinates"] = df["coordinates"].apply(Point)
    df["location"] = location
    df["survey_date"] = pd.to_datetime(date_string, yearfirst=True, dayfirst=False, format="%Y%m%d")
    df["raw_date"] = date_string
    df["tr_id"] = int(line_id)
    gdf_rgb = gpd.GeoDataFrame(df, geometry="coordinates")

    # Last touch, the proj4 info (coordinate reference system) is gathered with
    # Geopandas and applied to the newly created one.
    gdf_rgb.crs = str(transect_file.crs)

    # Let's create unique IDs from the coordinates values, so that the Ids
    # follows the coordinates
    gdf_rgb["point_id"] = [
        create_id(gdf_rgb.iloc[i]) for i in range(0, gdf_rgb.shape[0])
    ]

    if bool(add_xy):
        # Adding long/lat fields
        gdf_rgb["x"] = gdf_rgb.coordinates.x
        gdf_rgb["y"] = gdf_rgb.coordinates.y
    else:
        pass

    return gdf_rgb


def extract_from_folder(
    dataset_folder,
    transect_folder,
    tr_ids,
    list_loc_codes,
    mode,
    sampling_step,
    add_xy=False,
    add_slope=False,
    default_nan_values=-10000
):
    """Wrapper to extract profiles from all rasters inside a folder.

    Warning: The folders must contain the geotiffs and geopackages only.

    Args:
        dataset_folder (str): Path of the directory containing the datasets (geotiffs, .tiff).
        transect_folder (str): Path of the directory containing the transects (geopackages, .gpkg).
        tr_ids (str): If 'reset', a new incremental transect_id will be automatically assigned. If the name of a column in the transect files is provided, use that column as transect IDs.
        list_loc_codes (list): list of strings containing location codes.
        mode (str): If 'dsm', extract from DSMs. If 'ortho', extracts from orthophotos.
        sampling_step (float): Distance along-transect to sample points at. In meters.
        add_xy (bool): If True, adds extra columns with long and lat coordinates in the input CRS.
        add_slope (bool): If True, computes slope raster in degrees (increased processing time) and extract slope values across transects.
        default_nan_values (int): Value used for NoData in the raster format. In Pix4D, this is -10000 (Default).

    Returns:
        gdf (gpd.GeoDataFrame): A geodataframe with survey and topographical or color information extracted.
    """

    # Get a list of all the filenames and path
    list_files = filter_filename_list(
        getListOfFiles(dataset_folder), fmt=[".tif", ".tiff"]
    )

    dates = [getDate(dsm_in) for dsm_in in list_files]

    # List all the transects datasets
    if os.path.isdir(transect_folder):
        list_trans = getListOfFiles(transect_folder)
    elif os.path.isfile(transect_folder):
        list_trans = getListOfFiles(transect_folder)

    start = time.time()

    # Set the sampling distance (step) for your profiles

    gdf = pd.DataFrame()
    counter = 0

    if bool(add_slope):
        warnings.warn(
            "WARNING: add_terrain could increase processing time considerably for fine scale DSMs."
        )

    for dsm in tqdm(list_files):
        with ras.open(dsm, 'r') as ds:
            nan_values = ds.nodata
            if nan_values:
                pass
            else:
                nan_values=default_nan_values

        date_string = getDate(dsm)
        location = getLoc(dsm, list_loc_codes)


        if bool(add_slope):

            terr = rd.LoadGDAL(dsm, no_data=nan_values)
            print(
                f"Computing slope DSM in degrees in {location} at date: {date_string} . . ."
            )
            slope = rd.TerrainAttribute(terr, attrib="slope_degrees")
        else:
            slope = False

        transect_file_input = [a for a in list_trans if location in a]
        transect_file = gpd.read_file(transect_file_input[0])

        tr_list = np.arange(0, transect_file.shape[0])
        for i in tqdm(tr_list):
            if mode == "dsm":
                temp = get_profiles(
                    dsm=dsm,
                    transect_file=transect_file,
                    tr_ids=tr_ids,
                    transect_index=i,
                    step=sampling_step,
                    location=location,
                    date_string=date_string,
                    add_xy=add_xy,
                    add_terrain=slope,
                )
            elif mode == "ortho":
                temp = get_profile_dn(
                    ortho=dsm,
                    transect_file=transect_file,
                    transect_index=i,
                    step=sampling_step,
                    location=location,
                    tr_ids=tr_ids,
                    date_string=date_string,
                    add_xy=add_xy,
                )

            gdf = pd.concat([temp, gdf], ignore_index=True)

        counter += 1

    if counter == len(list_files):
        print("Extraction successful")
    else:
        print(f"There is something wrong with this dataset: {list_files[counter]}")

    end = time.time()
    timepassed = end - start

    print(
        f"Number of points extracted:{gdf.shape[0]}\nTime for processing={timepassed} seconds\nFirst 10 rows are printed below"
    )

    if mode == "dsm":
        nan_out = np.count_nonzero(np.isnan(np.array(gdf.z).astype("f")))
        nan_raster = np.count_nonzero(gdf.z == nan_values)
        gdf.z.replace(-10000, np.nan, inplace=True)

    elif mode == "ortho":
        nan_out = np.count_nonzero(
            np.isnan(np.array(gdf[["band1", "band2", "band3"]]).astype("f"))
        )
        nan_raster = np.count_nonzero(gdf.band1 == nan_values)
        gdf.band1.replace(0.0, np.nan, inplace=True)
        gdf.band2.replace(0.0, np.nan, inplace=True)
        gdf.band3.replace(0.0, np.nan, inplace=True)

    print(
        f"Number of points outside the raster extents: {nan_out}\nThe extraction assigns NaN."
    )
    print(
        f"Number of points in NoData areas within the raster extents: {nan_raster}\nThe extraction assigns NaN."
    )

    return gdf


def get_lod_table(multitemp_data, alpha=0.05):

    alpha=0.05
    multitemp_data['dh_abs']=[abs(i) for i in multitemp_data.dh]

    means=multitemp_data.groupby(["location","dt"]).dh.apply(np.mean)
    meds=multitemp_data.groupby(["location","dt"]).dh.apply(np.median).reset_index()
    nmads=multitemp_data.groupby(["location","dt"]).dh.apply(median_abs_deviation, **{'scale':'normal'})
    stds=multitemp_data.groupby(["location","dt"]).dh.apply(np.std)
    a_q683 = multitemp_data.groupby(["location","dt"]).dh_abs.apply(np.quantile, **{'q':.683})
    a_q95 = multitemp_data.groupby(["location","dt"]).dh_abs.apply(np.quantile, **{'q':.95})

    lod_stats=pd.DataFrame({'location':meds.location,
                 'dt': meds.dt,
                            'mean':means.values,
                  'med':meds.dh.values,
                 'std':stds.values,
                 'nmad': nmads.values,
                 'a_q683':a_q683.values,
                 'a_q95': a_q95.values})

    lod_stats["rrmse"]= np.sqrt(lod_stats.med**2 +  lod_stats.nmad**2)

    df_long=pd.DataFrame()

    for loc in multitemp_data.location.unique():
        data_loc_in=multitemp_data.query(f"location=='{loc}'")

        for dt_i in data_loc_in.dt.unique():
            data_lod = data_loc_in.query(f"dt=='{dt_i}'").dh
            data_lod.dropna(inplace=True)

            mean = np.mean(data_lod, axis=0)
            sd = np.std(data_lod, axis=0)
            data_out = [x for x in data_lod if x < (3 * sd)]
            data_out = [x for x in data_lod if -x < (3 * sd)]
            n_out=len(data_lod) - len(data_out)

            saphiro_stat, saph_p = shapiro(data_lod)
            ago_stat, ago_p = normaltest(data_lod)

            if saph_p > alpha:
                saphiro_normality='normal'
            else:
                saphiro_normality='not normal'

            if ago_p > alpha:
                ago_normality='normal'
            else:
                ago_normality='not normal'


            df_tmp=pd.DataFrame({'location':loc,
                                'dt':dt_i}, index=[0])

            df_tmp["n"]=len(data_lod)
            df_tmp["n_outliers"]=n_out
            df_tmp["saphiro_stat"]=saphiro_stat
            df_tmp["saphiro_p"]=saph_p
            df_tmp["ago_stat"]=ago_stat
            df_tmp["ago_p"]=ago_p
            df_tmp["saphiro_normality"]=saphiro_normality
            df_tmp["ago_normality"]=ago_normality

            df_long=pd.concat([df_tmp,df_long], ignore_index=True)

    lod_df=pd.merge(lod_stats,df_long)
    lod_df['lod']=np.where(np.logical_or(lod_df['saphiro_normality'] == 'normal',lod_df['ago_normality'] == 'normal'),
            lod_df['std'], lod_df['nmad'])

    return lod_df

def plot_lod_normality_check(multitemp_data, lod_df, details_table, locations,alpha=0.05,xlims=None,ylim=None,qq_xlims=None,qq_ylims=None,figsize=(7,4)):


    if isinstance(locations, list):
        if len(locations)>=1:
            loc_list=locations
        else:
            raise ValueError("Locations list passed is empty!.")
    elif locations=='all':
        loc_list=multitemp_data.location.unique()

    else:
        raise ValueError("Locations parameter must be a list of location codes or 'all'.")

    for loc in loc_list:

        data_loc_in=multitemp_data.query(f"location=='{loc}'")

        for dt_i in data_loc_in.dt.unique():

            lod_df_selection=lod_df.query(f"location=='{loc}' & dt == '{dt_i}'")

            f, (ax1,ax2) = plt.subplots(nrows=1,ncols=2, figsize=figsize)

            specs=details_table.query(f"location=='{loc}' & dt=='{dt_i}'")
            full_loc=specs.location.values[0]
            date_from=specs.date_pre.values[0]
            date_to=specs.date_post.values[0]
            dt=specs.dt.values[0]

            data_lod = data_loc_in.query(f"dt=='{dt_i}'").dh
            data_lod.dropna(inplace=True)

            mean = lod_df_selection["mean"].iloc[0]
            sd = lod_df_selection["std"].iloc[0]
            nmad = lod_df_selection["nmad"].iloc[0]

            data_out = [x for x in data_lod if x < (3 * sd)]
            data_out = [x for x in data_lod if -x < (3 * sd)]

            n_out=lod_df_selection["n_outliers"].iloc[0]
            saphiro_stat, saph_p = lod_df_selection["saphiro_stat"].iloc[0],lod_df_selection["saphiro_p"].iloc[0]
            ago_stat, ago_p = lod_df_selection["ago_stat"].iloc[0],lod_df_selection["ago_p"].iloc[0]


            f.suptitle(f"{full_loc} - {date_from} to {date_to} ({dt})")

            ax1.set_title(f'Density histograms ({len(lod_df_selection)} check points)')
            ax1.set_ylabel('Density')
            ax1.set_xlabel('Δh (m AHD)')

            if isinstance(xlims, tuple):
                ax1.set_xlim(*xlims)
            if ylim != None:
                ax1.set_ylim(ylim)

            ax1.axvline(nmad, color='red')
            ax1.axvline(sd, color='blue')

            dist=sb.histplot(data_out,kde=False, ax=ax1, stat='probability',
                              line_kws=dict(edgecolor="w", linewidth=1)
                                 )

            ax1.grid(b=None,axis="x")
            ax1.tick_params(axis="x", rotation=90)
            ax1.tick_params(axis="x")
            ax1.tick_params(axis="y")


            a=qqplot(abs(pd.Series(data_out)), line='s', fit=True, ax=ax2)
            ax2.set_xlabel('Theoretical quantiles')
            ax2.set_ylabel('Δh quantiles')

            if isinstance(qq_xlims, tuple):
                ax2.set_xlim(*qq_xlims)
            if isinstance(qq_ylims, tuple):
                ax2.set_ylim(*qq_ylims)

            ax2.set_title('Q-Q plot of absolute Δh')

            ax2.tick_params(axis="x")
            ax2.tick_params(axis="y")

            ax1.annotate(f"nmad: {np.round(nmad,2)}",color="red",xycoords="axes fraction", xy=(0.03, 0.97), xytext=(0.03, 0.97))
            ax1.annotate(f"std: {np.round(sd,2)}",color="blue",xycoords="axes fraction", xy=(0.03, 0.93), xytext=(0.03, 0.93))
            ax1.annotate(f"3σ outliers: {n_out}",color="k",xycoords="axes fraction", xy=(0.03, 0.89), xytext=(0.03, 0.89))

            saph_txt=f"Saphiro-Wilk: W = {np.round(saphiro_stat,2)}, p = < 0.05"
            ago_txt=f"D'Agostino-Pearson: K2 = {np.round(ago_stat,2)}, p = < 0.05"

            ax2.annotate(saph_txt,color="k",xycoords="axes fraction", xy=(0.03, 0.97), xytext=(0.03, 0.97))
            ax2.annotate(ago_txt,color="k",xycoords="axes fraction", xy=(0.03, 0.93), xytext=(0.03, 0.93))

            if saph_p > alpha:
                conc_txt_saph='Saphiro-Wilk --> Normal distribution.'
            else:
                conc_txt_saph='Saphiro-Wilk --> Non-normal distribution.'


            if ago_p > alpha:
                conc_txt_ago="D'Agostino-Pearson --> Normal distribution."
            else:
                conc_txt_ago="D'Agostino-Pearson --> Non-normal distribution."

            ax2.annotate(conc_txt_saph,color="k",xycoords="axes fraction", xy=(0.10, 0.2), xytext=(0.10, 0.08))
            ax2.annotate(conc_txt_ago,color="k",xycoords="axes fraction", xy=(0.10, 0.035), xytext=(0.10, 0.035))

            dots = a.findobj(lambda x: hasattr(x, 'get_color') and x.get_color() == 'b')
            line = a.findobj(lambda x: hasattr(x, 'get_color') and x.get_color() == 'r')
            [d.set_markersize(1) for d in dots]
            [d.set_alpha(0.3) for d in dots]
            [d.set_color('k') for d in dots]
            line[0].set_color('k')
            line[0].set_ls('--')

            ax1.grid(axis='y')
            ax1.grid(b=None,axis='x')

def get_rbcd_transect(df_labelled, loc_specs, reliable_action, dirNameTrans, labels_order, loc_codes, crs_dict_string):
    """It computes the r-BCDs at the transect level, based on the timeseries of elevation change magnituteds across the beachface dataset (markov_tag dataframe).

    Args:
        df_labelled (pd.DataFrame): Pandas dataframe with dh magnitude labelled.
        loc_specs (dict): Dictionary where keys are the location codes and values are location-specific (inner) dictionaries where keys are 'thresh' and 'min_points' and values are the associated values, like loc_specs={'mar':{'thresh':6,'min_points':6}, 'leo':{'thresh':4,'min_points':20}}.
        reliable_action (str) : Insert "drop" (default) to remove transects that have less than the specified number of non-nnn points (thresh) or "keep" to keep them.
        dirNameTrans (str): Path of the directory containing the transects (geopackages, .gpkg).
        labels_order (list): Order of labels (magnitude of change) to be preserved.
        loc_codes (list): List of strings containing location codes.
        crs_dict_string (dict): Dictionary storing location codes as key and crs information as values, in dictionary form.

    Returns:
       rbcd_transects (gpd.GeoDataFrame): A GeoDataFrames containing the steady-state distribution of each transect.
    """
    steady_state_tr = pd.DataFrame()

    df_labelled["spatial_id"]=[spatial_id(geometry_in) for geometry_in in df_labelled.geometry]

    for loc in tqdm(df_labelled.location.unique()):
        data_loc = df_labelled.query(f"location=='{loc}'")

        if loc not in loc_specs.keys():
            print(f"No threshold and mi_points provided in loc_specs for {loc}. Using no filters.")
            loc_thresh=0
            loc_min_pts=0

        else:
            loc_thresh=loc_specs[loc]['thresh']
            loc_min_pts=loc_specs[loc]['min_points']

        for tr_id in data_loc.tr_id.unique():

            data_tr = data_loc.query(f"tr_id=='{tr_id}'")

            if data_tr.empty:
                data_tr = data_loc.query(f"tr_id=={tr_id}")
                data_tr["spatial_id"]=[spatial_id(geometry_in) for geometry_in in data_tr.geometry]

            data_piv = data_tr.pivot(
                values='markov_tag',
                index="spatial_id",
                columns='dt',
            )

            # identify the points that have less than the required number of transitions (thresh) of non nan states
            valid_pts=((~data_piv.isnull()).sum(axis=1)>=loc_thresh).sum()

            # has this transect a number of valid points above the specified min_pts parameter?
            valid_transect= valid_pts >= loc_min_pts

            # drop ivalid points
            data_piv.dropna(axis=0, thresh=loc_thresh, inplace=True)

            # all the  NaN will be named 'nnn'
            data_piv.fillna("nnn", inplace=True)

            n = data_piv.shape[0]
            arr = np.array(data_piv)
            m = Markov(arr)

            try:
                steady_state = m.steady_state
                steady_state = pd.DataFrame(
                    m.steady_state, index=m.classes, columns=[tr_id]
                )

                steady_state.reset_index(inplace=True)
                steady_state.rename({"index": "markov_tag"}, axis=1, inplace=True)
                steady_state = steady_state.melt(
                    id_vars="markov_tag", value_name="p", var_name="tr_id"
                )
                steady_state["location"] = loc
                steady_state["thresh"] = loc_thresh
                steady_state["min_pts"] = loc_min_pts
                steady_state["valid_pts"] = valid_pts
                steady_state["reliable"] = valid_transect

                steady_state_tr = pd.concat(
                    [steady_state, steady_state_tr], ignore_index=True
                )

            except BaseException:
                print(f"tr_id {tr_id} has {n} valid points.")
                null_df=pd.DataFrame({'markov_tag':df_labelled.markov_tag.unique(),
                                    'p':[np.nan for i in df_labelled.markov_tag.unique()]})
                null_df["tr_id"]=tr_id
                null_df["location"]=loc
                null_df["thresh"]=loc_thresh
                null_df["min_pts"]=loc_min_pts
                null_df["valid_pts"]=valid_pts
                null_df["reliable"]=valid_transect

                steady_state_tr = pd.concat(
                    [null_df, steady_state_tr], ignore_index=True
                )

    # what to do with unreliable transects?
    if reliable_action =='drop':
        steady_state_tr=steady_state_tr.query("reliable == True")
    elif reliable_action =='keep':
        pass
    else:
        raise ValueError("The parameter 'reliable_action' must be either 'drop' or 'keep'.")

    ss_transects_idx = pd.DataFrame()

    idx_matrix=len(labels_order)//2

    for loc in steady_state_tr.location.unique():

        sub = steady_state_tr.query(f"location=='{loc}'")
        sub = sub.pivot(index="markov_tag", columns="tr_id", values=("p"))

        if len(sub.index)!=len(labels_order):
            raise ValueError(f" The following magnitude labels are missing from the steady state dataset: {set(sub.index).symmetric_difference(set(labels_order))}.")
        else:
            pass

        sub = sub.loc[labels_order, :]

        # Create erosion and deposition sub-matrix
        erosion = sub.iloc[idx_matrix:, :].transpose()
        erosion["erosion"] = erosion.sum(axis=1)
        erosion=erosion.reset_index()[["tr_id","erosion"]]

        deposition = sub.iloc[:-idx_matrix, :].transpose()
        deposition["deposition"] = deposition.sum(axis=1)
        deposition=deposition.reset_index()[["tr_id","deposition"]]
        merged_erodepo=pd.merge(erosion, deposition)

        merged_erodepo["residual"] = merged_erodepo.deposition - merged_erodepo.erosion
        merged_erodepo["location"] = loc

        to_plot = merged_erodepo.melt(
            id_vars=["tr_id"], var_name="process", value_name="coastal_index"
        )
        to_plot["location"] = loc

        path_trs=glob.glob(rf"{dirNameTrans}/{loc}*")[0]
        transect_in = gpd.read_file(path_trs)
        transect_in.columns= transect_in.columns.str.lower()

        merged_erodepo["geometry"] = pd.merge(
                    merged_erodepo, transect_in[["tr_id","geometry"]], how="left", on="tr_id"
                ).geometry

        ss_transects_idx_loc = gpd.GeoDataFrame(
                        merged_erodepo, geometry="geometry", crs=crs_dict_string[loc]
                    )
        ss_transects_idx=pd.concat([ss_transects_idx_loc,ss_transects_idx], ignore_index=True)

    return ss_transects_idx



def get_coastal_Markov(arr_markov, weights_dict, store_neg=True):
    """Compute BCDs from first-order transition matrices of dh magnitude classes (as states).

    Args:
        arr_markov (np.array): Numpy array of markov transition matrix.
        weights_dict (dict): Dictionary with keys:dh classes, values: weight (int). Especially useful for the e-BCDs magnitude trend (sign).
        store_neg (bool): If True (default), use the subtraction for diminishing trends.

    Returns:
        (tuple): Tuple containing:
            BCD(float) the actual BCD indices
            trend(float) value of the indices trend
            sign(str) can be '-' or '+' for plotting purposes.
    """

    combs = pd.Series(product(arr_markov.index, (arr_markov.columns)))

    value_trans = 0
    value_trend = 0

    for state1, state2 in combs:

        state_1_w = weights_dict[state1]  # extract the weights
        state_2_w = weights_dict[state2]
        value = arr_markov.at[state1, state2]

        if state_1_w > state_2_w:  # check if the change is decr

            if bool(store_neg):
                weigth_adhoc_trend = state_1_w * (-(state_2_w))

            value_trans += value
            value_trend += value * weigth_adhoc_trend

        elif state_1_w < state_2_w:
            weigth_adhoc_trend = state_1_w * state_2_w

            value_trans += value
            value_trend += value * weigth_adhoc_trend

        else:
            weigth_adhoc_trend = state_1_w * state_2_w

            value_trans += value
            value_trend += value * weigth_adhoc_trend

    if value_trend > 0:
        sign = "+"
    elif value_trend < 0:
        sign = "-"
    elif value_trend == 0:
        sign = "0"
    else:
        sign = np.nan

    return np.round(value_trans, 3), np.round(value_trend, 3), sign



def compute_multitemporal (df,
                            geometry_column="coordinates",
                           date_field='survey_date',
                           filter_class='sand'):
    """
    From a dataframe containing the extracted points and a column specifying whether they are sand or non-sand, returns a multitemporal dataframe
    with time-periods sand-specific elevation changes.

    Args:
        date_field (str): The name of the column storing the survey date.
        geometry_column (str): Name of the column containing the geometry.
        filter_classes (str,list): Name of the class of list of classes to be retained in the multitemporal computation.

    Returns:
        multiteporal_df (pd.DataFrame): A multitemporal dataframe of sand-specific elevation changes.
    """
    if filter_class != None:
        # check if pt_class in columns
        if "pt_class" not in df.columns:
            raise ValueError("The data is not classified as no 'pt_class' column is present. Please run the method cleanit on the ProfileDynamics object first.")
        else:
            if isinstance(filter_class, str):
                filter_classes_in=[filter_class]
            elif isinstance(filter_class, list):
                filter_classes_in=filter_class
            else:
                raise ValueError(" If provided, class_filter must be either a string or a list of strings containing the classes to retain.")

        print(f"Filter activated: only {filter_classes_in} points will be retained.")
    else:
        filter_classes_in=["no_filters_applied"]
        pass

    df["spatial_id"]=[create_spatial_id(df.iloc[i]) for i in range(df.shape[0])]
    fusion_long=pd.DataFrame()


    for location in df.location.unique():
        print(f"working on {location}")
        loc_data=df.query(f"location=='{location}'")
        list_dates=loc_data.loc[:,date_field].unique()
        list_dates.sort()


        for i in tqdm(range(list_dates.shape[0])):

            if i < list_dates.shape[0]-1:
                date_pre=list_dates[i]
                date_post=list_dates[i+1]
                print(f"Calculating dt{i}, from {date_pre} to {date_post} in {location}.")

                if filter_class != None:
                    df_pre=loc_data.query(f"{date_field} =={date_pre} & pt_class in {filter_classes_in}").dropna(subset=['z'])
                    df_post=loc_data.query(f"{date_field} =={date_post} & pt_class in {filter_classes_in}").dropna(subset=['z'])
                else:
                    df_pre=loc_data.query(f"{date_field} =={date_pre}").dropna(subset=['z'])
                    df_post=loc_data.query(f"{date_field} =={date_post}").dropna(subset=['z'])

                merged=pd.merge(df_pre,df_post, how='inner', on='spatial_id', validate="one_to_one",suffixes=('_pre','_post'))
                merged["dh"]=merged.z_post.astype(float) - merged.z_pre.astype(float)

                dict_short={"geometry":merged.filter(like=geometry_column).iloc[:,0],
                            "location":location,
                            "tr_id":merged.tr_id_pre,
                            "distance":merged.distance_pre,
                            "class_filter":'_'.join(filter_classes_in),
                            "dt":  f"dt_{i}",
                            "date_pre":date_pre,
                            "date_post":date_post,
                            "z_pre":merged.z_pre.astype(float),
                            "z_post":merged.z_post.astype(float),
                            "dh":merged.dh}

                short_df=pd.DataFrame(dict_short)
                fusion_long=pd.concat([short_df,fusion_long],ignore_index=True)

    print("done")
    return fusion_long

def sensitivity_tr_rbcd(ProfileDynamics,
                       test_thresholds='max',
                       test_min_pts=[0,10,2]):
    """Performs a sensitivity analysis of the transect-level r-BCDs values in respect to both the min_points and thresh parameters.

    Args:
        ProfileDynamics (object): The ProfileDynamics object to perform sensitivity analysis on.
        test_thresholds (str, list): If a list is provided, the list must contain the starting thresh value, the last one and the interval between the two, just like numpy.arange function expects, like [0,10,2]. If 'max', then tests on all the available timeperiods.
        test_min_pts (list): A list containing the starting min_pts value, the last one and the interval between the two, just like numpy.arange function expects, like [0,100,10].

    Returns:
        ss_tr_big (pd.DataFrame): A dataframe storing the transect r-BCD values of all combinations of thresh and min_points.
    """
    df = ProfileDynamics.df_labelled
    ss_tr_big=pd.DataFrame()

    for loc in df.location.unique():
        data_in=df.query(f"location=='{loc}'")
        print(f"Working on {loc}.")

        if test_thresholds=='max':
            range_thresh=range(0,data_in.dt.unique().shape[0]+1)

        else:
            range_thresh=range(*test_thresholds)

        if test_min_pts==None:
            range_min_pts=range(0,20,2)
        else:
            range_min_pts=range(*test_min_pts)

        combs = list(itertools.product(range_min_pts,range_thresh))
        print(f"A total of {len(combs)} combinations of thresholds and min_pts will be computed.")

        for i in tqdm(combs):
            print(f"Working on threshold {i[1]} and min points {i[0]}.")
            tmp_loc_specs_dict={loc:{'thresh':i[1],
                            'min_points':i[0]}}

            try:
                ss_transects_idx = get_rbcd_transect(df_labelled=data_in,
                      loc_specs=tmp_loc_specs_dict, reliable_action='drop',
                      dirNameTrans=ProfileDynamics.ProfileSet.dirNameTrans,
                      labels_order=ProfileDynamics.tags_order,
                      loc_codes=ProfileDynamics.ProfileSet.loc_codes,
                      crs_dict_string=ProfileDynamics.ProfileSet.crs_dict_string)

                ss_transects_idx['thresh']=i[1]
                ss_transects_idx['min_pts']=i[0]

                ss_tr_big=pd.concat([ss_tr_big,ss_transects_idx], ignore_index=True)
            except:
                print("errore")

                pass

    return ss_tr_big

def plot_sensitivity_rbcds_transects(df, location, x_ticks=[0,2,4,6,8],figsize=(7,4),
                                     tresh_xlims=(0,8), trans_ylims=(0,3), sign_ylims=(0,10)):
    """Plot both the number of valid transects retained (red) and the total sign changes (blue) as a function of the threshold used. This function creates a plot per min_points situation. The solid black line is the 95th percentile of the total valid transect retained while the dashed one is the 85th.

    Args:
        df (pd.DataFrame): Sensitivity dataframe resulting from using the function sensitivity_tr_rbcd().
        location (str): Location code of the locatin to plot.
        x_ticks (list): List of x-axis ticks (thresholds).
        figsize (tuple): Width and height (in inches) of the figure.
        tresh_xlims (tuple): Min x and max x of the thresholds x-axis.
        trans_ylims (tuple): Min y and max y of the transect main y-axis.
        sign_ylims (tuple): .
    """

    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['font.family'] = 'sans-serif'
    sb.set_context("paper", font_scale=1)

    q_up_val=0.95
    q_low_val=0.85


    data_in=df.query(f"location == '{location}'")

    list_minpts=data_in.min_pts.unique()
    trs_res_ar=data_in.groupby(["tr_id","min_pts"])['residual'].apply(np.array).reset_index()
    tot_trs=data_in.groupby(["thresh","min_pts"])['geometry'].count().reset_index()
    tot_trs['trs_10']=tot_trs.geometry / 10
    zero_crossings=pd.DataFrame([pd.Series({'tr_id':trs_res_ar.loc[i,'tr_id'],
                                            'sign_change_thresh':np.where(np.diff(np.sign(trs_res_ar.iloc[i,-1])))[0][-1]+1,
                                           'min_pts':trs_res_ar.loc[i,'min_pts']}) for i in range(trs_res_ar.shape[0]) if np.where(np.diff(np.sign(trs_res_ar.iloc[i,-1])))[0].shape[0] !=0])
    tot_jumps=zero_crossings.groupby(["sign_change_thresh","min_pts"]).count().reset_index() # how many jumps per thresh and minpts

    joined=pd.merge(tot_trs,tot_jumps, left_on=['thresh','min_pts'], right_on=['sign_change_thresh','min_pts'], how='left')
    joined.rename({'geometry':'tot_trs',
                  'tr_id':'tot_jumps'}, axis=1, inplace=True)


    for minpts in list_minpts:

        f,ax=plt.subplots(figsize=figsize)
        ax2=ax.twinx()

        datain=joined.query(f"min_pts=={minpts}")


        sb.lineplot(x="thresh", y="tot_jumps",ci=None,
                        data=datain,color='b',
                       alpha=.4,linewidth=3,
                    ax=ax2, label="sign changes")

        sb.lineplot(data=datain,x='thresh',y='trs_10',
                    alpha=.4,color='r',linewidth=3,
                    ax=ax,label="transects * 10")


        kde_x, kde_y = ax.lines[0].get_data()
        kde_x2, kde_y2 = ax2.lines[0].get_data()
        ax.fill_between(kde_x, kde_y,interpolate=True, color='r',alpha=0.5)
        ax2.fill_between(kde_x2, kde_y2,interpolate=True,color='b',alpha=0.5)

        ax.axhline((datain.tot_trs.fillna(0).max()*q_up_val)/10,c='k',ls='-',label='95%')
        ax.axhline((datain.tot_trs.fillna(0).max()*q_low_val)/10,c='k',lw=2.5,ls='--',label='85%')

        ax.set_ylabel('n. transects x 10', c='r')
        ax.set_xlabel('t')
        ax2.set_ylabel('sign changes', c='b')
        ax2.set_ylim(sign_ylims)
        ax.set_ylim(trans_ylims)
        ax.set_xlim(tresh_xlims)


        plt.tight_layout()
        ax.get_legend().remove()
        ax2.get_legend().remove()


        plt.xticks(x_ticks)

        ax.set_title(f"pt: {minpts}")
        plt.tight_layout()


def get_sil_location(merged_df, ks, feature_set, random_state=10):
    """
    Function to obtain average Silhouette scores for a list of number of clusters (k) in all surveys.
    It uses KMeans as a clusteres and parallel processing for improved speed.

    Warning:
        It might take up to 8h for tables of 6,000,000 observations.

    Args:
        merged_df (pd.DataFrame): The clean and merged dataframe containing the features.
        ks (tuple): starting and ending number of clusters to run KMeans and compute SA on.
        feature_set (list): List of strings of features in the dataframe to use for clustering.
        random_state (int): Random seed used to make the randomisation deterministic.

    Returns:
        sil_df (pd.DataFrame): A dataframe containing average Silhouette scores for each survey, based on the provided feature set.
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
        opt_k (dict): Dictionary with optimal k for each survey.
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
            ):  # if multiple plateau values are found, obtain the mean k of those values

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
        ks (int, dict): number of clusters (k) or dictionary containing the optimal k for each survey. See get_opt_k function.
        feature_set (list): List of names of features in the dataframe to use for clustering.
        thresh_k (int): Minimim k to be used. If survey-specific optimal k is below this value, then k equals the average k of all above threshold values.
        random_state (int): Random seed used to make the randomisation deterministic.

    Returns:
        data_classified (pd.DataFrame): A dataframe containing the label_k column, with point_id, location, survey_date and the features used to cluster the data.
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


def prep_heatmap(df, lod, outliers=False, sigma_n=3):
    """
    Function to create a pivoted and filtered dataframe from multitemporal table of specific period-location combination (i.e. loc= pfa, dt= dt_3).
    Elevation differences within LoD (uncertain) can be set to zero and outliers can be eliminated.
    Each column is a transect and each row is a distance value along transect. Values are elevation differences.

    Warning:
        This function is to be used on location-period specific slices of the multitemporal table.

    Args:
        df (Pandas dataframe): Location-period specific subset (filtered for a location and a timeperiod) of multitemporal table.
        lod (path, value, False): if valid path to an Limit of Detection table, use the table. If a value is provided, use the value across all surveys. If False, do not apply LoD filter. All elevation changes within +- LoD will be set to zero.
        outliers: when True, use the specified number of standard deviation to exclude outliers. If False, retain all the points.
        sigma_n (int): number of standard deviation to use to exclude outliers (default=3).

    Returns:
        piv_df (pd.DataFrame): Pivoted and clean Pandas Dataframe, ready to be used to plot heatmaps and compute volumetrics.
    """

    # pivoting and sorting

    # add an int version of tr_id to sort better
    df["int_tr"] = df.tr_id.apply(lambda a: int(float(a)), convert_dtype=True)
    df.rename({"distance": "dist"}, axis=1, inplace=True)
    df.dist.astype(float)
    df.sort_values(by=["int_tr", "dt", "dist"], inplace=True)  # sort
    df_piv = df.pivot("dist", "int_tr", "dh")
    df_piv = df_piv.astype(float)

    df_piv.sort_index(axis=1, inplace=True)  # sort columns, tr_id
    df_piv.sort_index(axis=0, ascending=False, inplace=True)  # sort rows, distance

    if bool(outliers):
        # eliminating outliers

        # find threshold for sigma outlier detection
        thresh = sigma_n * np.nanstd(df_piv.astype(float).values)

        df_piv = df_piv[df_piv < thresh]  # select only the values below the threshold
        # select only the negative values below the threshold
        df_piv = df_piv[-df_piv < thresh]

    else:
        pass

    if isinstance(lod, str):
        if os.path.isfile(lod):  # if a valid path, ope the table and use it

            # applying LoDs

            lod_table = pd.read_csv(lod)  # read in the lod table
            loc = df.location.iloc[0]
            dt = df.dt.iloc[0]
            lod_i = np.round(lod_table.query(f"location == '{loc}' & dt == '{dt}'").lod.values[0],2)
            # create condition (within LoD) to mask the dataframe
            cond = (df_piv >= -lod_i) & (df_piv <= lod_i)
            # replace the values that satisfied the condition (within LoD) with zeroes
            df_piv2 = df_piv.mask(cond, 0)
            df_piv2.set_index(df_piv2.index.astype(float), inplace=True)

            return df_piv2.sort_index(ascending=False)
        else:
            raise NameError(
                "Not a valid file or path.Please provide a valid path to the LoD table"
            )

    elif isinstance(
        lod, (float, int)
    ):  # if a numeric, use this value across all surveys

        lod_i = float(lod)
        cond = (df_piv >= -lod_i) & (df_piv <= lod_i)
        df_piv2 = df_piv.mask(cond, 0)
        df_piv2.set_index(df_piv2.index.astype(float), inplace=True)

        return df_piv2.sort_index(ascending=False)

    elif isinstance(lod, pd.DataFrame):
        lod_table=lod
        loc = df.location.iloc[0]
        dt = df.dt.iloc[0]
        lod_i = np.round(lod_table.query(f"location == '{loc}' & dt == '{dt}'").lod.values[0],2)
        # create condition (within LoD) to mask the dataframe
        cond = (df_piv >= -lod_i) & (df_piv <= lod_i)
        # replace the values that satisfied the condition (within LoD) with zeroes
        df_piv2 = df_piv.mask(cond, 0)
        df_piv2.set_index(df_piv2.index.astype(float), inplace=True)


        return df_piv2.sort_index(ascending=False)
    else:  # otherwise,don't use it

        df_piv2 = df_piv.copy()
        df_piv2.set_index(df_piv2.index.astype(float), inplace=True)

        return df_piv2.sort_index(ascending=False)


def fill_gaps(data_in, y_heat_bottom_limit, spacing, bottom=True, y_heat_start=0):
    """Function to fill the pivoted table (returned from prep_heatmap function) with missing across-shore distances, due to align data on heatmaps. Empty rows (NaN) will be added on top (from 0 to the first valid distance) and, optionally on the bottom of each transect, (from the last valid distance to a specified seaward limit).

    Warning:
        This function assume along-transect distances to be going from land to water, which is not what the profiles distances represent originally.

    Args:
        data_in (pd.DataFrame): Pivoted dataframe, where each column is a transect and row is a along-shore distance.
        y_heat_bottom_limit (int): Lower boundary distance to extend all transects.
        bottom (bool): If True (default), rows are extended seaward too, up to y_heat_bottom_limit. If False, only distances from 0 to the first valid values will be added.
        y_heat_start (int): Landward starting distance value (default=0)
        spacing (float): Sampling step (meters) used to extract points (default=0.1)

    Returns:
        complete_df (pd.DataFrame): Complete dataframe with extra rows of NaN added.
    """
    if y_heat_bottom_limit < data_in.index[-1]:
        raise ValueError(f"y_heat_bottom_limit ({y_heat_bottom_limit}) cannot be lower than the maximum distance already present in the data ({data_in.index[-1]}).")
    if bool(bottom):
        bottom_fill_array = np.empty(
            (
                (
                    int(
                        np.round(y_heat_bottom_limit + spacing - data_in.index[-1], 1)
                    )
                ),
                data_in.shape[1],
            )
        )
        bottom_fill_array[:] = np.NaN
        to_concat_after = pd.DataFrame(
            data=bottom_fill_array,
            index=np.arange(data_in.index[-1], y_heat_bottom_limit + spacing, spacing),
            columns=data_in.columns,
        )
    else:
        pass

    before_fill_array = np.empty((int(data_in.index[0]), data_in.shape[1]))
    before_fill_array[:] = np.NaN
    to_concat_before = pd.DataFrame(
        data=before_fill_array,
        index=np.arange(y_heat_start, data_in.index[0], spacing),
        columns=data_in.columns,
    )

    if bool(bottom):
        return pd.concat([to_concat_before, data_in, to_concat_after.iloc[1:]])
    else:
        return pd.concat([to_concat_before, data_in])


def interpol_integrate(series, dx):
    """
    Linearly interpolate NaN values (non-sand) within the first and last valid points (from the swash to the landward end of each transect),
    and intergrate the area below this interoplated profile, to obtain transect specific estimates of volumetric change.
    Args:
        series (pd.Series): Series of elevation change with distance as indices.
        dx (int,float): The along-transect point spacing.
    Returns:
        integrated_area (float): Volumetric change in cubic meters.
    """
    min_dist, max_dist = (
        series.first_valid_index(),
        series.last_valid_index(),
    )  # get distances of first and last sand points

    interpol = series.loc[min_dist:max_dist].interpolate()  # interpolate linearly

    area_simps = simps(interpol.values, dx=dx)  # intergrate using Simpson's method

    return area_simps


def get_beachface_length(series):
    """Get across-shore beachface length from series of elevation change with distance as indices.

    Args:
        series (pd.Series): series of elevation change with distance as indices.

    Returns:
        beachface_length (float): Across-shore beachface length in meters.
    """

    min_dist, max_dist = series.first_valid_index(), series.last_valid_index()

    across_shore_beachface_length = np.round(max_dist - min_dist, 1)

    return across_shore_beachface_length


def get_m3_m_location(data_in, dx, transect_spacing=20):
    """
    Get alongshore-shore net volumetric change in cubic meters per meter of beach.

    Args:
        data_in (pd.Dataframe): Dataframe generated by prep_heatmap function.
        transect_spacing (int,float): Spacing between transects in meters.
        dx (int,float): The along-transect point spacing
    Returns:
        m3_m (float): Cubic meters of change per meter of beach alongshore, at the site level.
    """

    # compute alongshore beachface length
    along_beach = data_in.shape[1] * transect_spacing

    tot_vol = sum(
        (data_in.apply(interpol_integrate, axis=0, **{'dx':dx})) * transect_spacing
    )  # compute net volume change

    return tot_vol / along_beach  # return m3_m alongshore


def get_state_vol_table(
    sand_pts, lod, full_specs_table, dx, transect_spacing=20, outliers=False, sigma_n=3
):
    """
    Function to compute location-level altimetric beach change statistics from the multitemporal table.
    By default, only sand points beyond LoD are accounted for. Optionally, LoD filter can be turned off.
    The table contains info on:
    - monitoring period: location code, full name, period code, dates, number of days and valid points
    - absolute altimetric change: total beachface rising, lowering and net change
    - normalised altimetric change: meters of rising, lowering and net elevation change per valid survey point (MEC)


    Args:
        sand_pts (Pandas dataframe): multitemporal table.
        lod (str, bool): if valid path to an LoD table, use the table. If a value is provided, use the value across all surveys. If False, do not apply LoD filter.cAll elevation change (dh) values within LoD will be set to zero.
        full_specs_table (False, path): Full path to the table with extended monitoring info. If False, monitoring period information are limited.
        dx (int,float): The along-transect point spacing.
        transect_spacing (int): Alongshore spacing of transects (m)
        outliers (bool): when True, use the specified number of standard deviation to exclude outliers. If False, retain all the points.
        sigma_n (int): number of standard deviation to use to exclude outliers (default=3).


    Returns:
        volumetrics_location (pd.DataFrame): A dataframe storing altimetric beach change info and other information for every period and location.
    """

    if isinstance(full_specs_table, bool):
        if bool(full_specs_table) == False:
            skip_details = True
        else:
            print("Please provide the path to the specs table.")

    elif isinstance(full_specs_table, str):
        if os.path.isfile(full_specs_table):
            table_details = pd.read_csv(full_specs_table)
            skip_details = False
        else:
            raise TypeError("Not a valid path to the .csv file for the specs table.")

    elif isinstance(full_specs_table, pd.DataFrame):
        table_details = full_specs_table
        skip_details = False

    else:
        raise TypeError("Not a valid path to the .csv file for the specs table.")


    tr_df_full = pd.DataFrame()

    locs = sand_pts.location.unique().tolist()

    for loc in locs:
        test_loc = sand_pts.query(f"location == '{loc}'")

        for dt in test_loc.dt.unique():
            test_dt = test_loc.query(f" dt =='{dt}'")
            data_in = prep_heatmap(test_dt, lod=lod, outliers=outliers, sigma_n=sigma_n)

            # compute erosion and deposition volumes at site level
            data_in_erosion = data_in[data_in < 0]
            data_in_deposition = data_in[data_in > 0]

            if bool(skip_details):
                pass
            else:
                specs = table_details.query(f"location=='{loc}' & dt=='{dt}'")
                full_loc = specs.loc_full.values[0]
                date_from = specs.date_pre.values[0]
                date_to = specs.date_post.values[0]
                n_days = specs.n_days.values[0]

            beach_length = len(data_in.columns) * transect_spacing

            n_obs_valid = (
                data_in.count().sum()
            )  # sand only, within beachface, LoD filtered (default)

            abs_in = data_in[data_in > 0].sum().sum()  # total profiles rising
            abs_out = data_in[data_in < 0].sum().sum()  # total profiles lowering
            abs_net_change = data_in.sum().sum()  # net altimetric change

            mec_m = (
                abs_net_change / beach_length
            )  # meters of elevation change per meter of beach

            norm_in = abs_in / n_obs_valid  # meters of profile rising per valid point
            norm_out = (
                abs_out / n_obs_valid
            )  # meters of profile lowering per valid point
            norm_net_change = abs_net_change / n_obs_valid  # MEC

            tot_vol_depo = (
                data_in_deposition.apply(interpol_integrate, axis=0, **{'dx':dx}) * transect_spacing
            ).sum()
            tot_vol_ero = (
                data_in_erosion.apply(interpol_integrate, axis=0, **{'dx':dx}) * transect_spacing
            ).sum()
            net_vol_change = tot_vol_depo + tot_vol_ero
            location_m3_m = net_vol_change / beach_length

            if bool(skip_details) == False:
                df_dict = {
                    "location": loc,
                    "location_full": full_loc,
                    "dt": dt,
                    "date_from": date_from,
                    "date_to": date_to,
                    "n_days": n_days,
                    "abs_in": abs_in,
                    "abs_out": abs_out,
                    "abs_net_change": abs_net_change,
                    "mec_m": mec_m,
                    "norm_in": norm_in,
                    "norm_out": norm_out,
                    "norm_net_change": norm_net_change,
                    "tot_vol_depo": tot_vol_depo,
                    "tot_vol_ero": tot_vol_ero,
                    "net_vol_change": tot_vol_depo - abs(tot_vol_ero),
                    "location_m3_m": location_m3_m,
                    "n_obs_valid": n_obs_valid,
                }
            else:
                df_dict = {
                    "location": loc,
                    "dt": dt,
                    "abs_in": abs_in,
                    "abs_out": abs_out,
                    "abs_net_change": abs_net_change,
                    "mec_m": mec_m,
                    "norm_in": norm_in,
                    "norm_out": norm_out,
                    "norm_net_change": norm_net_change,
                    "tot_vol_depo": tot_vol_depo,
                    "tot_vol_ero": tot_vol_ero,
                    "net_vol_change": tot_vol_depo - abs(tot_vol_ero),
                    "location_m3_m": location_m3_m,
                    "n_obs_valid": n_obs_valid,
                }

            df = pd.DataFrame(df_dict, index=[0])

            tr_df_full = pd.concat([df, tr_df_full], ignore_index=True)

    return tr_df_full


def get_transects_vol_table(
    sand_pts,
    lod,
    dx,
    full_specs_table,
    transect_spacing=20,  # INTEGRATED
    outliers=False,
    sigma_n=3,
):
    """
    Function to compute transect-level altimetric change statistics from the multitemporal table.
    By default, only sand points beyond LoD are accounted for. Optionally, LoD filter can be turned off.
    The table contains info on:
    - monitoring period: location code, full name, period code, transect ID, dates, number of days and valid points
    - absolute altimetric change: total profile rising, lowering and net change
    - normalised altimetric change: meters of rising, lowering and net elevation change per valid survey point (MEC)

    Args:
        sand_pts (pd.DataFrame): Multitemporal dh table.
        lod (str, float, False): If valid path to an LoD table, use the table. If a value is provided, use the value across all surveys. If False, do not apply LoD filter. All elevation change (dh) values within LoD will be set to zero.
        dx (int,float): The along-transect point spacing.
        full_specs_table (str, pd.DataFRame, bool): Full path to the table with extended monitoring info or the table itself. If False, monitoring period information are limited.
        outliers: when True, use the specified number of standard deviation to exclude outliers. If False, retain all the points.
        sigma_n (int): number of standard deviation to use to exclude outliers (default=3).

    Returns:
        volumetrics_transects (pd.DataFrame): A dataframe storing altimetric beach change info at the transect level, and other information for every period and location.
    """

    transects_df_full = pd.DataFrame()

    locs = sand_pts.location.unique().tolist()

    if isinstance(full_specs_table, bool):
        if bool(full_specs_table) == False:
            skip_details = True
        else:
            print("Please provide the path to the specs table.")

    elif isinstance(full_specs_table, str):
        if os.path.isfile(full_specs_table):
            table_details = pd.read_csv(full_specs_table)
            skip_details = False
        else:
            raise TypeError("Not a valid path to the .csv file for the specs table.")

    elif isinstance(full_specs_table, pd.DataFrame):
        table_details = full_specs_table
        skip_details = False

    else:
        raise TypeError("Not a valid path to the .csv file for the specs table.")


    for loc in locs:

        test_loc = sand_pts.query(f"location == '{loc}'")

        for dt in test_loc.dt.unique():
            test_dt = test_loc.query(f" dt =='{dt}'")
            data_in = prep_heatmap(test_dt, lod, outliers=False, sigma_n=3)

            tr_ids = sorted(data_in.columns.values)

            if skip_details != True:
                specs = table_details.query(f"location=='{loc}' & dt=='{dt}'")
                full_loc = specs.loc_full.values[0]
                date_from = specs.date_pre.values[0]
                date_to = specs.date_post.values[0]
                n_days = specs.n_days.values[0]
            else:
                pass

            trs_volumes = data_in.apply(interpol_integrate, **{'dx':dx}, axis=0)
            trs_volumes.name = "tot_vol_change"
            beach_lengths = data_in.apply(get_beachface_length, axis=0)

            tr_df = pd.DataFrame(trs_volumes)
            # normalise the volume change computed in the transect by its cross-shore
            # length
            tr_df["m3_m"] = (
                trs_volumes.values * transect_spacing
            ) / beach_lengths.values
            tr_df = tr_df.reset_index()
            tr_df.rename({"int_tr": "tr_id"}, axis=1, inplace=True)

            tr_df["n_obs_valid"] = data_in.count().values
            tr_df["abs_in"] = data_in[data_in > 0].sum().values
            tr_df["abs_out"] = data_in[data_in < 0].sum().values
            tr_df["abs_net_change"] = data_in.sum().values

            tr_df["mec_m"] = tr_df.abs_net_change.values / beach_lengths.values

            tr_df["norm_in"] = tr_df.abs_in.values / tr_df.n_obs_valid.values
            tr_df["norm_out"] = tr_df.abs_out.values / tr_df.n_obs_valid.values
            tr_df["norm_net_change"] = (
                tr_df.abs_net_change.values / tr_df.n_obs_valid.values
            )

            tr_df["dt"] = dt
            tr_df["location"] = loc

            if skip_details != True:
                tr_df["location_full"] = full_loc
                tr_df["date_from"] = date_from
                tr_df["date_to"] = date_to
                tr_df["n_days"] = n_days

            else:
                pass

            transects_df_full = pd.concat([tr_df, transects_df_full], ignore_index=True)

    return transects_df_full


def plot_alongshore_change(
    sand_pts,
    mode,
    lod,
    full_specs_table,
    return_data=False,
    location_subset=["wbl"],
    dt_subset=["dt_0"],
    ax2_y_lims=(-8, 5),
    save=False,
    save_path="C:\\your\\preferred\\folder\\",
    dpi=300,
    img_type=".png",
    from_land=True,
    from_origin=True,
    add_orient=False,
    fig_size=(7.3, 3),
    font_scale=1,
    plots_spacing=0,
    bottom=False,
    y_heat_bottom_limit=80,
    transect_spacing=20,
    along_transect_sampling_step=1,
    heat_yticklabels_freq=5,
    heat_xticklabels_freq=5,
    outliers=False,
    sigma_n=3,
):
    """
    Display and optionally save alongshore altimetric and volumetric beach changes plots.
    A subset of locations and periods can be plotted.
    If LoD parameter is True (default), then white cells in the altimetric heatmap are values within LoD. Grey cells is no data or no sand points.
    Optionally, LoD filter can be turned off.

    Args:
        sand_pts (Pandas dataframe): multitemporal table.
        mode (str): if 'subset', only a subset of locations and dts are plotted. If 'all', all periods and locations are plotted. .
        lod (path, value, False): if valid path to an LoD table, use the table. If a value is provided, use the value across all surveys. If False, do not apply LoD filter. All elevation change (dh) values within LoD will be set to zero.
        full_specs_table (False, path): Full path to the table with extended monitoring info. If False, monitoring period information are limited.
        location_subset (list): list of strings containing the location codes (e.g. wbl) to be plotted.
        dt_subset (list): list of strings containing the period codes (e.g. dt_0) to be plotted.
        ax2_y_lims (tuple): limits of y-axis of alonghsore volumetric change plot. Default is (-8,5).
        save (bool): If True, saves the plots in the specified save_path. False is default.
        save_path (path): Full path to a folder (e.g. C:\\preferred\\folder\\) where to save plots.
        dpi (int): Resolution in Dot Per Inch (DPI) to save the images.
        img_type (str): '.png','.pdf', '.ps', '.svg'. Format of the saved figures.
        from_land (bool): If True (default), cross-shore distances are transformed into landward distances, where 0 is the end of beachface.
        from_origin (bool): If True (default), transect IDS are transformed in alongshore distance from origin (tr_id=0). It requires regularly spaced transects.
        add_orient (bool): if True, an additional lineplot is added to the volumetric plot containing orientation info. It needs pre-computed orientations (tr_orient parameter) (TO UPDAte). False is default.
        fig_size (tuple): Tuple of float to specify images size. Default is (7.3,3).
        font_scale (float): Scale of text. Default=1.
        plots_spacing (float): Vertical spacing of the heatmap and alongshore change plots. Default = 0.
        bottom (bool): If True (default), rows are extended seaward too, up to y_heat_bottom_limit. If False, only distances from 0 to the first valid values will be added.
        y_heat_bottom_limit (int): Lower boundary distance (seaward) to extend all transects to.
        transect_spacing (float): Alongshore spacing of transects (m).
        outliers (bool): when True, use the specified number of standard deviation to exclude outliers. If False, retain all the points.
        sigma_n (int): number of standard deviation to use to exclude outliers (default=3).

    """

    sb.set_context("paper", font_scale=font_scale, rc={"lines.linewidth": 0.8})

    if isinstance(full_specs_table, bool):
        if bool(full_specs_table) == False:
            skip_details = True
        else:
            raise TypeError(
                "Not a DataFrame, nor a valid path to the .csv file for the specs table."
            )
    elif isinstance(full_specs_table, str):
        if os.path.isfile(full_specs_table):
            table_details = pd.read_csv(full_specs_table)
            skip_details = False
        else:
            TypeError(
                "The path provided in full_spec_table is not a valid path to the .csv file."
            )
    elif isinstance(full_specs_table, pd.DataFrame):
        table_details = full_specs_table
        skip_details = False
    else:
        raise TypeError(
            "Not a DataFrame, nor a valid path to the .csv file for the specs table."
        )

    land_limits = pd.DataFrame(
        sand_pts.groupby(["location"]).distance.max()
    ).reset_index()

    locations_to_analyse = sand_pts.location.unique()

    if bool(from_origin):
        xlabel = ""
    else:
        xlabel = "Transect ID"

    if mode == "subset":
        locations_to_analyse = location_subset
        dt_to_analyse = dt_subset

    elif mode == "all":
        locations_to_analyse = sand_pts.location.unique()
        dt_to_analyse = sand_pts.query(f"location == '{loc}'").dt.unique()

    for loc in locations_to_analyse:
        temp_loc = sand_pts.query(f"location == '{loc}'")

        for dt in dt_to_analyse:

            # subset the data
            temp = sand_pts.query(f"location == '{loc}' and dt =='{dt}'")

            # prepare axes and figure
            f, (ax, ax2) = plt.subplots(
                nrows=2,
                figsize=fig_size,
                sharex=True,
                gridspec_kw={"hspace": plots_spacing},
            )

            # _____________data_preparation______________

            # compute beach length based on number of transects
            beach_length = len(temp.tr_id.unique()) * transect_spacing

            # prepare the data to be suitable for the heatmap
            data_in = prep_heatmap(temp, lod=lod, outliers=outliers, sigma_n=sigma_n)

            if bool(skip_details):
                full_loc = loc
                lod_i = 0.05
            else:

                specs = table_details.query(f"location=='{loc}' & dt=='{dt}'")
                full_loc = specs.loc_full.values[0]
                date_from = specs.date_pre.values[0]
                date_to = specs.date_post.values[0]
                n_days = specs.n_days.values[0]

            if isinstance(lod, pd.DataFrame):
                lod_i = np.round(lod.query(f"location == '{loc}' & dt == '{dt}'").lod.values[0],2)
            elif isinstance(lod, (float, int)):
                lod_i = lod
            else:
                lod_i = 0.05

            # FIGURE_______________________________________________________________________

            #
            axins = ax.inset_axes(bounds=[1.02, 0, 0.04, 1])

            if bool(from_land):

                land_lim = land_limits.query(f"location=='{loc}'").distance.values[0]
                data_in["m_from_land"] = np.round(land_lim - data_in.index, 1)
                data_in.set_index("m_from_land", inplace=True)
            else:
                pass

            if bool(from_origin):
                data_in.columns = data_in.columns.astype("int") * transect_spacing
            else:
                pass

            data_in_filled = fill_gaps(data_in, y_heat_bottom_limit, bottom=bottom, spacing=along_transect_sampling_step)
            print(f"Working on {loc} at {dt}")

            # _______AX__________________________________________________________________

            sb.heatmap(
                data=data_in_filled,
                xticklabels=heat_xticklabels_freq,
                facecolor="w",
                robust=True,
                center=0,
                ax=ax,
                cbar_kws={"label": "Δh m AHD"},
                cbar=True,
                cbar_ax=axins,
                cmap="seismic_r",
                vmin=-0.8,
                vmax=0.8,
            )

            # _________________________________BACKGROUND COLOR AND TRANSPARENCY_____
            ax.patch.set_facecolor("grey")
            ax.patch.set_alpha(0.4)

            # _________________________________AXIS LABELS_____
            ax.set_xlabel("")
            ax.set_ylabel("Cross. distance (m)")
            ax.set_title(f"")

            # __________________________________SPINES,TICKS_AND_GRIDS_________________________________
            ax.get_xaxis().set_visible(True)
            ax.spines["bottom"].set_visible(True)
            ax.spines["left"].set_visible(True)
            ax.spines["right"].set_visible(True)
            ax.spines["top"].set_visible(True)

            ax.grid(b=True, which="minor", linewidth=0.3, ls="-")
            ax.grid(b=True, which="major", linewidth=0.8, ls="-")

            tmp_list_along_dists = np.arange(0, y_heat_bottom_limit + 5, heat_yticklabels_freq)
            y_formatter_list_values = tmp_list_along_dists.astype("str").tolist()
            y_formatter_list_values[-1] = ""
            y_formatter_list_locators = [value * (1/along_transect_sampling_step) for value in tmp_list_along_dists]

            y_formatter = FixedFormatter(y_formatter_list_values)
            y_locator = FixedLocator(y_formatter_list_locators)

            ax.yaxis.set_major_formatter(y_formatter)
            ax.yaxis.set_major_locator(y_locator)
            ax.yaxis.set_minor_locator(AutoMinorLocator(4))
            ax.grid(which="minor", axis="y")

            ax.set_ylim(y_heat_bottom_limit * (1/along_transect_sampling_step), 0)
            ax.tick_params(axis="x", which="both", length=0)

            # __AX2___________________________________________________________________

            red_patch = mpatches.Patch(color="orange", label="Erosion")
            blue_patch = mpatches.Patch(color="skyblue", label="Deposition")

            trs_volumes = data_in.apply(interpol_integrate, axis=0, **{'dx':along_transect_sampling_step})
            beach_lengths = data_in.apply(get_beachface_length, axis=0)

            m3_m = (trs_volumes * transect_spacing) / beach_lengths

            trs_volumes.reset_index().rename({"int_tr": "tr_id"}, axis=1, inplace=True)
            trs_volumes.name = "dh"
            tr_df = pd.DataFrame(trs_volumes)

            tr_df.reset_index(inplace=True)
            tr_df["net_volume_change_m3"] = tr_df.dh
            tr_df["net_balance_m3_m"] = m3_m.values

            tr_df.set_index("int_tr")
            tr_df.reset_index(inplace=True)

            sb.lineplot(data=tr_df, x="index", y="net_balance_m3_m", ax=ax2, color="k")
            ax2.set_ylim(ax2_y_lims)
            ax2.yaxis.set_minor_locator(AutoMinorLocator(4))

            # FILLS________________________________________

            line_x, line_y = ax2.lines[0].get_data()

            ax2.fill_between(
                line_x,
                line_y,
                where=(line_y > 0),
                interpolate=True,
                color="skyblue",
                alpha=0.3,
            )

            ax2.fill_between(
                line_x,
                line_y,
                where=(line_y < 0),
                interpolate=True,
                color="orange",
                alpha=0.3,
            )

            ax2.grid(b=True, which="minor", linewidth=0.5, ls="-")
            ax2.grid(b=True, which="major", linewidth=0.8, ls="-")

            ax2.set_xlabel("Along. distance (m)")
            ax2.set_ylabel("Net Δv (m³/m)")
            ax2.axhline(0, color="k", ls="--", zorder=1)
            ax2.set_title(f"")

            ax2.spines["top"].set_visible(False)
            ax2.spines["right"].set_visible(True)
            ax2.spines["left"].set_visible(True)
            ax2.spines["bottom"].set_visible(True)
            ax2.grid(which="minor", axis="y")

            if bool(skip_details) == False:
                date_from_str = pd.to_datetime(str(date_from)).strftime("%d %b '%y")
                date_to_str = pd.to_datetime(str(date_to)).strftime("%d %b '%y")

                f.suptitle(
                    f"Beachface change in {full_loc} from {date_from_str} to {date_to_str} (LoD = {lod_i} m)"
                )
            else:

                f.suptitle(f"Beachface change in {loc} for {dt} (LoD = {lod_i} m)")

            axs = f.get_axes()

            f.align_ylabels(axs[:-1])

            ax.set_zorder(1)
            ax2.set_zorder(0)

            if bool(save):
                savetxt = save_path + loc + "_" + dt + "_AlongChange" + img_type
                f.savefig(savetxt, dpi=dpi, bbox_inches="tight")
            else:
                pass

            plt.show()

    if return_data:
        return data_in_filled


def plot_mec_evolution(
    volumetrics,
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
    dpi=300,
    img_type=".png",
    save_fig=False,
    name_fig=f"Mean Elevation Changes",
    save_path=None,
):

    pd.options.mode.chained_assignment = None  # default='warn'

    sb.set_context("paper", font_scale=font_scale)
    myFmt = DateFormatter("%d.%m.%y")

    # sort the locations
    if loc_order:

        sorterIndex = dict(zip(loc_order, range(len(loc_order))))
        volumetrics["loc_rank"] = volumetrics["location"].map(sorterIndex)
        volumetrics.sort_values(["loc_rank", "dt"], inplace=True)

    else:
        pass

    volumetrics["date_from_dt"]=[datetime.datetime.strptime(str(pre),'%Y%m%d') for pre in volumetrics.loc[:, date_from_field]]
    volumetrics["date_to_dt"]=[datetime.datetime.strptime(str(post),'%Y%m%d') for post in volumetrics.loc[:, date_to_field]]

    num_subplots = volumetrics.location.unique().shape[0]
    if num_subplots > 1:

        fig, axs = plt.subplots(
            1,
            num_subplots,
            sharey=True,
            sharex=False,
            squeeze=True,
            figsize=figure_size,
        )

        for ax_i, loc in zip(axs.flatten(), volumetrics.location.unique()):

            data_in = volumetrics.query(f"location=='{loc}'")
            data_in.sort_values("date_to_dt", inplace=True)
            # add the cumulative change curve
            data_in["cum_change"] = data_in.norm_net_change.cumsum()

            first_date_from = data_in.loc[:, "date_from_dt"].iloc[0]

            full_loc = data_in.loc[:, location_field].iloc[0]
            x = data_in.norm_net_change
            x2 = data_in.cum_change
            y = dates.date2num(data_in.loc[:, "date_to_dt"])
            y_scatter = dates.date2num(data_in.loc[:, "date_from_dt"])

            # LinePlots

            # normalised change
            ax_i.plot(x, y, lw=0.8, c="k")

            # cumulative norm.change
            ax_i.plot(x2, y, lw=0.8, ls="--", c="k", zorder=4)

            # from start of monitoring to first "date_to"
            x_start = np.array([0, x.iloc[0]])
            y_start = [y_scatter[0], y[0]]

            ax_i.plot(x_start, y_start, lw=1, c="k", zorder=3)

            # Vertical Line
            ax_i.axvline(0, color="grey", ls="-", lw=0.8, zorder=0)

            # Spines
            ax_i.spines["right"].set_visible(False)
            ax_i.spines["left"].set_visible(False)
            ax_i.spines["top"].set_visible(False)
            ax_i.spines["bottom"].set_linewidth(0.8)
            ax_i.spines["bottom"].set_color("k")

            # Grid
            ax_i.grid(color="grey", linestyle="-", linewidth=0.8, alpha=0.3)

            # ScatterPlots
            ax_i.scatter(x, y, c="k", s=5, zorder=5)
            ax_i.scatter(0, y_start[0], c="k", s=5, zorder=5)

            # ticks
            if scale_mode == "auto":

                abs_range = abs(max(x) - (min(x)))

                if abs_range < 0.2:
                    tick_step = 0.05

                else:
                    tick_step = 0.1

                ticks_value = np.arange(
                    round_special(min(x), 0.05),
                    round_special(max(x), 0.05) + tick_step,
                    tick_step,
                )
                ax_i.set_xticks(ticks_value)

            elif scale_mode == "equal":
                if isinstance(x_diff, dict):

                    if loc in x_diff.keys():
                        print(
                            f"x_diff provided. Setting xlims of {loc} = {x_diff[loc][0],x_diff[loc][1]} "
                        )
                        ax_i.set_xlim(x_diff[loc][0], x_diff[loc][1])
                    else:
                        print(
                            f"x_diff provided but {loc} not found. Setting xlims= {x_limits} "
                        )
                        ax_i.set_xlim(x_limits)
                else:
                    ax_i.set_xlim(x_limits)

            start, end = ax_i.get_ylim()
            ax_i.yaxis.set_ticks(np.arange(start, end, dates_step))

            # SubPlot Title
            ax_i.set_title(f"{full_loc}")
            ax_i.yaxis.grid(True)
            ax_i.yaxis.set_major_formatter(myFmt)

        for ax in fig.axes:
            plt.sca(ax)
            plt.locator_params(axis="x", nbins=x_binning)

            plt.xticks(rotation=90)

    else:

        fig, ax = plt.subplots(figsize=figure_size)

        data_in = volumetrics
        data_in.sort_values("date_to_dt", inplace=True)
        # add the cumulative change curve
        data_in["cum_change"] = data_in.norm_net_change.cumsum()

        first_date_from = data_in.date_from_dt.iloc[0]

        full_loc = data_in.loc[:, location_field].iloc[0]
        x = data_in.norm_net_change
        x2 = data_in.cum_change
        y = dates.date2num(data_in.date_to_dt)
        y_scatter = dates.date2num(data_in.date_from_dt)

        # LinePlots

        # normalised change
        ax.plot(x, y, lw=0.8, c="k")

        # cumulative norm.change
        ax.plot(x2, y, lw=0.8, ls="--", c="k", zorder=4)

        # from start of monitoring to first "date_to"
        x_start = np.array([0, x.iloc[0]])
        y_start = [y_scatter[0], y[0]]

        ax.plot(x_start, y_start, lw=1, c="k", zorder=3)

        # Vertical Line
        ax.axvline(0, color="grey", ls="-", lw=0.8, zorder=0)

        # Spines
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_linewidth(0.8)
        ax.spines["bottom"].set_color("k")

        # Grid
        ax.grid(color="grey", linestyle="-", linewidth=0.8, alpha=0.3)

        # ScatterPlots
        ax.scatter(x, y, c="k", s=5, zorder=5)
        ax.scatter(0, first_date_from, c="k", s=5, zorder=5)

        # ticks
        if scale_mode == "optimised":

            abs_range = abs(max(x) - (min(x)))

            if abs_range < 0.2:
                tick_step = 0.05

            else:
                tick_step = 0.1

            ticks_value = np.arange(
                round_special(min(x), 0.05),
                round_special(max(x), 0.05) + tick_step,
                tick_step,
            )
            ax.set_xticks(ticks_value)

        elif scale_mode == "equal":  # Inverloch different scale, for display purposes
            if loc != "inv":

                ax.set_xlim(x_limits)
            else:
                ax.set_xlim(-1.1, 1.1)

        else:
            pass

        start, end = ax.get_ylim()
        ax.yaxis.set_ticks(np.arange(start, end, dates_step))

        # SubPlot Title
        ax.set_title(f"{full_loc}")
        ax.yaxis.grid(True)
        ax.yaxis.set_major_formatter(myFmt)

    for ax in fig.axes:
        plt.sca(ax)
        plt.locator_params(axis="x", nbins=x_binning)

        plt.xticks(rotation=90)

    fig.tight_layout(pad=0.2)

    if bool(save_fig):

        savetxt = save_path + name_fig + img_type

        plt.savefig(savetxt, dpi=dpi)
    else:
        pass


def plot_single_loc(
    df,
    loc_subset,
    figsize,
    colors_dict,
    linewidth,
    out_date_format,
    xlabel,
    ylabel,
    suptitle,
):
    """
    Display Mean Elevation Change (mec, in m) and cumulative mec (since start of the monitoring), for a single location.

    Args:
        df (Pandas dataframe): location-level-volumetrics table obtained from get_state_vol_table function.
        loc_subset (list): a list of location codes, in case multiple locations need to be plotted (not optimal).
        figsize (tuple): Tuple of float to specify images size.
        colors_dict (dict): Dictionary with keys=location code and values=color (in matplotlib specification).
        linewidth (float, int): linewidths of the plot lines.
        out_date_format (str): format of the dates plotted in the x axis (datetime format).
        xlabel (str): labels for x axis.
        ylabel (str): labels for y axis.
        suptitle: title of the plot.

    """
    f, ax = plt.subplots(figsize=figsize)

    if isinstance(colors_dict, dict):
        print("Color dictionary provided.")
        color_mode = "dictionary"

    elif colors_dict == None:

        num_colors = len(loc_subset)
        cmapa = plt.cm.get_cmap("tab20c")
        cs = [cmapa(i) for i in np.linspace(0, 0.5, num_colors)]
        color_mode = "auto"

    else:
        raise TypeError("Error in specifying color dictionary.")

    dataset_in = df.query(f"location in {loc_subset}")
    dataset_in["date_from_dt"] = pd.to_datetime(dataset_in.date_from, format="%Y%m%d", dayfirst=False)
    dataset_in["date_to_dt"] = pd.to_datetime(dataset_in.date_to, format="%Y%m%d", dayfirst=False)

    for i, location in enumerate(dataset_in.location.unique()):

        dataset = dataset_in.query(f"location=='{location}'")


        if color_mode == "dictionary":
            color_i = colors_dict[location]
        elif color_mode == "auto":
            color_i = cs[i]

        # Calculate the cumulative net volumetric change and mean elevation change from start of monitoring
        dataset["cum"] = dataset.net_vol_change.cumsum()
        dataset["cum_mec"] = dataset.norm_net_change.cumsum()


        ax = sb.lineplot(
            data=dataset,
            x="date_to_dt",
            color=color_i,
            y="net_vol_change",
            label=f"{location}: period",
            lw=linewidth,
        )
        ax = sb.lineplot(
            data=dataset,
            x="date_to_dt",
            color=color_i,
            y="cum",
            label=f"{location}: cumulative",
            linestyle="--",
            lw=linewidth,
        )

        ax = sb.scatterplot(
            data=dataset, color=color_i, x="date_to_dt", y="net_vol_change"
        )
        ax = sb.scatterplot(data=dataset, color=color_i, x="date_to_dt", y="cum")

        x_start = np.array([dataset.iloc[0].date_from_dt, dataset.iloc[0].date_to_dt])
        y_start = np.array([0, dataset.iloc[0].cum])

        ax.plot(x_start, y_start, c=color_i)
        ax.scatter(dataset.iloc[0].date_from_dt, 0, c=color_i, marker="*")

    # the zero horizontal line
    ax.axhline(0, c="k", lw=0.5)

    ax.set(
        xticks=np.append(dataset_in.date_from_dt.values, dataset_in.date_to_dt.values[-1])
    )
    ax.xaxis.set_major_formatter(dates.DateFormatter(out_date_format))

    plt.xticks(rotation=90)

    # the plot x and y axis labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    # the title of the plot
    f.suptitle(suptitle)

    return ax


def images_to_dirs(images_folder, target_folder, op=None):
    """Create one folder per image named as the image. Optionally, move image into it.

    Args:
        images_folder (str): path to the directory where the images are stored.
        target_folder (str): target path where create subfolders named as the images.
        op (None, "copy", "move"): Move or copy the images into the newly created folders. None, creates empty subfolders.
    """

    images = os.listdir(images_folder)
    original_images_paths = [os.path.join(images_folder, image) for image in images]
    ids = [os.path.splitext(image)[0] for image in images]

    if op != None:
        target_images_paths = [os.path.join(target_folder, id_in) for id_in in ids]
    # create ID-folders in target_parent_folder, named as image names

    starting_wd = os.getcwd()  # store starting working directory

    os.chdir(target_folder)  # change working dir

    if op == None:
        for id_i in ids:
            if not os.path.exists(os.path.join(target_folder, id_i)):
                os.mkdir(id_i)
            else:
                print(f"{id_i} exists already. Skipping this one.")

    elif op != None:

        for id_i, source, destination in zip(
            ids, original_images_paths, target_images_paths
        ):
            if not os.path.exists(os.path.join(target_folder, id_i)):
                os.mkdir(id_i)
                if op == "move":
                    move(source, destination)
                elif op == "copy":
                    copy(source, destination)
                else:
                    raise ValueError(
                        "op parameter not set. 'move' to move images into newly created folders, 'copy' to copy them or None to only create folders."
                    )

            else:
                print(f"{id_i} exists already. {op} only.")
                if op == "move":
                    move(source, destination)
                elif op == "copy":
                    copy(source, destination)
    else:
        raise ValueError(
            "op parameter not set. 'move' to move images into newly created folders, 'copy' to copy them or None to only create folders."
        )

    os.chdir(starting_wd)  # returning to starting working dir

    print(f"Successfully created {len(ids)} ID-folders in {target_folder} .")


def s2_to_rgb(imin, scaler_range=(0, 255), re_size=False, dtype=False):
    """Transform image pixels into specified range. Used to obtain 8-bit RGB images.

    Args:
        imin (np.array): image array to be transformed.
        scaler_range (tuple): (min,max) tuple with minimum and maximum brightness values. Defaults is 0-255.
        re_size (False, tuple): if a tuple of size 2 is provided, reshape the transformed array with the provided shape. Default to False.
        dtype (False, dtype): optional data type for the transformed array. False, keep the original dtype.
    Returns:
        img_array_rgb (np.array): Transformed array.
    """

    scaler = MinMaxScaler(scaler_range)

    imo = ras.open(imin, "r")
    if dtype != False:
        im = imo.read(out_dtype=dtype)
    else:
        im = imo.read()

    if isinstance(re_size, (tuple)):
        im = resize(
            im,
            (re_size[0], re_size[1], re_size[2]),
            mode="constant",
            preserve_range=True,
        )  # resize image
    else:
        pass

    if imo.count > 1:
        im_rgb = np.stack((im[0], im[1], im[2]), axis=-1)
        rgb_array_1 = b = im_rgb.reshape(-1, 1)
        scaled_1 = scaler.fit_transform(rgb_array_1).astype(int)
        img_array_rgb = scaled_1.reshape(im_rgb.shape)
    else:
        img_array_rgb = im

    if isinstance(re_size, (tuple)):
        img_array_rgb = resize(
            im,
            (re_size[0], re_size[1], re_size[2]),
            mode="constant",
            preserve_range=True,
        )  # resize image

    return img_array_rgb


def shoreline_from_prediction(
    prediction, z, shapely_affine, min_vertices=2, shape=(64, 64)
):
    """
    Extract a georeferenced subpixel shoreline from an array using Marching squares and store it in a GeoDataframe.
    Credits: adapted from Dr. Robbie Bishop-Taylor functions in Digital Earth Australia scripts, available at:
    https://github.com/GeoscienceAustralia/dea-notebooks/blob/develop/Scripts/dea_coastaltools.py

    Args:
        prediction (array): The 2D array returned by the DL model.
        z (float,int): The threshold to use to divide water and no-water predicted binary image.
        shapely_affine (Affine object): Shapely Affine object of the tile the prediction has been performed from.
        min_vertices (int): Minimum number of vertices to retain a shoreline segment (default=2).
        shape (tuple): Shape of the tiles (default= (64,64), minimum requirement for Unet).

    Returns:
        gpd.GeoDataFrame: Geodataframe containing the contours (shoreline) extracted from the prediction.
    """
    # get shoreline
    shore_arr = contours_to_multiline(
        prediction.reshape(shape), z, min_vertices=min_vertices
    )

    # create geoseries and geodataframe
    shore_arr_geoseries = gpd.GeoSeries(shore_arr, name="geometry")
    contours_gdf = gpd.GeoDataFrame(shore_arr_geoseries, geometry="geometry")

    # georeference line using tile geotransform
    contours_gdf["geometry"] = contours_gdf.affine_transform(shapely_affine)

    return contours_gdf


def grid_from_pts(pts_gdf, width, height, crs, offsets=(0, 0, 0, 0)):
    """
    Create a georeferenced grid of polygones from points along a line (shoreline).
    Used to extract tiles (images patches) from rasters.
    Args:
        pts_gdf (gpd.GeoDataFrame): The geodataframe storing points along a shoreline.
        width (int, float) The width of each single tile of the grid, given in the CRS unit (use projected CRS).
        height (int,float): The height of each single tile of the grid, given in the CRS unit (use projected CRS).
        crs (str): Coordinate Reference System in the dictionary format (example: {'init' :'epsg:4326'})
        offsets (tuple): Offsets in meters (needs projected CRS) from the bounds of the pts_gdf, in the form of (xmin, ymin, xmax, ymax). Default to (0,0,0,0).
    Returns:
        Grid (gpd.GeoDataFrame): A GeoDataFrame storing polygon grids, with IDs and geometry columns.
    """

    xmin, ymin, xmax, ymax = tuple(map(operator.add, pts_gdf.total_bounds, offsets))

    rows = int(np.ceil((ymax - ymin) / height))
    cols = int(np.ceil((xmax - xmin) / width))

    XleftOrigin = xmin
    XrightOrigin = xmin + width
    YtopOrigin = ymax
    YbottomOrigin = ymax - height
    polygons = []
    for i in range(cols):
        Ytop = YtopOrigin
        Ybottom = YbottomOrigin
        for j in range(rows):
            polygons.append(
                Polygon(
                    [
                        (XleftOrigin, Ytop),
                        (XrightOrigin, Ytop),
                        (XrightOrigin, Ybottom),
                        (XleftOrigin, Ybottom),
                    ]
                )
            )
            Ytop = Ytop - height
            Ybottom = Ybottom - height
        XleftOrigin = XleftOrigin + width
        XrightOrigin = XrightOrigin + width

    grid = gpd.GeoDataFrame(
        {"grid_id": range(len(polygons)), "geometry": polygons}, crs=crs
    )

    return grid


def add_grid_loc_coords(grid_gdf, location=None):
    """
    Add coordinate fields of the corners of each grid tiles.

    Args:
        grid_gdf (gpd.GeoDataFrame): The geodataframe storing the grid, returned by the grid_from_pts function.
        location (str): The location code associated with the grid. Defaults to None.

    Returns:
        grid_gdf (gpd.GeoDataFrame): The original grid, with UpperLeft X and Y (ulx,uly), UpperRight X and Y (urx,ury), LowerLeft X and Y (llx,llr) and LowerRigth X and Y (lrx,lry) coordinates fields added.
    """

    if location is None:
        location = np.nan

    ulxs = []
    urxs = []
    lrxs = []
    llxs = []

    ulys = []
    urys = []
    lrys = []
    llys = []

    for grid in range(grid_gdf.shape[0]):

        coords = grid_gdf.iloc[grid].geometry.exterior.coords.xy

        # get upper-left, upper-right, lower-right and lower-left X coordinates.
        ulx = coords[0][0]
        urx = coords[0][1]
        lrx = coords[0][2]
        llx = coords[0][3]

        # get upper-left, upper-right, lower-right and lower-left Y coordinates.
        uly = coords[1][0]
        ury = coords[1][1]
        lry = coords[1][2]
        lly = coords[1][3]

        ulxs.append(ulx)
        urxs.append(urx)
        lrxs.append(lrx)
        llxs.append(llx)

        ulys.append(uly)
        urys.append(ury)
        lrys.append(lry)
        llys.append(lly)

    grid_gdf.loc[:, "ulx"] = ulxs
    grid_gdf.loc[:, "urx"] = urxs
    grid_gdf.loc[:, "lrx"] = lrxs
    grid_gdf.loc[:, "llx"] = llxs

    grid_gdf.loc[:, "uly"] = ulys
    grid_gdf.loc[:, "ury"] = urys
    grid_gdf.loc[:, "lry"] = lrys
    grid_gdf.loc[:, "lly"] = llys

    grid_gdf.loc[:, "location"] = location

    return grid_gdf


def grid_from_shore(
    shore,
    width,
    height,
    location_code,
    adj_order=1,
    crs="shore",
    shore_res=10,
    offsets=(0, 0, 0, 0),
    plot_it=True,
):
    """
    Create a georeference grid of equal polygones (tiles) along a line (shoreline) and select those tiles that contain or are adjacent to the shoreline.

    Args:
        shore (geodataframe): The geodataframe storing the input line from where the grid will be created.
        width, height (int,float): The width and height of each single tile of the grid, given in the CRS unit (use projected CRS).
        location_code (str): The location code associated with the grid.
        adj_order (False, int): Contiguity order to subset grid cells adjacent to shoreline. If False, only cells
            directly touching the shoreline will be extracted (Default=1). Note: The Pysal Queen method is used to compute neighbors.
            For more info: https://pysal.org/libpysal/generated/libpysal.weights.Queen.html#libpysal.weights.Queen
        crs (dict or 'shore'): If 'shore', use the same CRS of the input line. If dict, keys should be the location code and values the values crs in the dictionary format ('wbl' : {'init' :'epsg:32754'}).
        shore_res (int,float): the alongshore spacing of points plotted along the line in the CRS unit (default=10). It doesn't need to be a small value, it is used to get the extent of the bounding box that encapsulate all the shoreline, before split this into a grid.
        offsets (tuple): Offsets in meters (needs projected CRS) from the bounds of the pts_gdf,
            in the form of (xmin, ymin, xmax, ymax). Default to (0,0,0,0).
        plot_it (bool): plot the shoreline, full grid and the tiles selected containing the lien (in red). Default to True.

    Returns:
        gpd.GeoDataFrame: Grid of tiles in the specified proximity of the shoreline.
    """

    xs = []
    ys = []
    points = []

    for distance in np.arange(
        0, shore.length.values[0], shore_res
    ):  # shore_res: meters alongshroe to get points from shore
        pt = shore.interpolate(distance)
        points.append(pt.values[0])
        xs.append(pt.values[0].x)
        ys.append(pt.values[0].y)

    pts = [[x, y] for x, y in zip(xs, ys)]
    pts = np.array(pts)

    if isinstance(crs, dict):
        crs_in = crs[location_code]
    elif crs == "shore":
        crs_in = shore.crs

    points_gdf = gpd.GeoDataFrame(
        {"local_id": range(len(points)), "geometry": points},
        geometry="geometry",
        crs=crs_in,
    )
    grid = grid_from_pts(points_gdf, width, height, crs=crs_in, offsets=offsets)

    # select grid cells that contains shoreline points
    shore_grids = grid[
        grid.geometry.apply(lambda x: points_gdf.geometry.intersects(x).any())
    ]

    if adj_order != False:
        w = Queen.from_dataframe(grid, "geometry")

        if adj_order >= 1:
            print(f"Higher order ({adj_order}) selected.")
            w = higher_order(w, adj_order)

            df_adjc = w.to_adjlist()

            # get the unique neighbors of all focal cells
            qee_polys_ids = set(
                df_adjc.query(f"focal in {list(shore_grids.grid_id)}").neighbor
            )

            # subset grid based on qee_polys_ids
            shore_grids = grid.query(f"grid_id in {list(qee_polys_ids)}")

        else:
            pass

    if bool(plot_it):
        f, ax = plt.subplots(figsize=(10, 10))

        grid.plot(color="white", edgecolor="black", ax=ax)
        shore.plot(ax=ax, color="b")
        shore_grids.geometry.boundary.plot(
            color=None, edgecolor="r", linewidth=1, ax=ax
        )

    add_grid_loc_coords(shore_grids, location=location_code)

    return shore_grids


def dissolve_shores(gdf_shores, field="date"):
    """Dissolves multi-part shorelines into one geometry per location-date. Uses GeoPandas.GeoDataFrame.dissolve method.

    args:
        gdf_shores (gpd.GeoDataFrame): The geodataframe storing the shoreline.
        field (str): The field to be used to dissolve shorelines. Default to "date".

    returns:
        dissolved (gpd.GeoDataFrame): GeoDataFrame with one geometry per location-date combination.
    """
    if len(gdf_shores.location.unique()) != 1:
        multi_location = True
    else:
        multi_location = False

    dissolved = pd.DataFrame()
    if bool(multi_location):
        for loc in gdf_shores.location.unique():
            geom = (
                gdf_shores.query(f"location=='{loc}'").dissolve(by=field).reset_index()
            )
            dissolved = pd.concat([dissolved, geom], ignore_index=True)
    else:
        gdf_shores["diss"] = 0
        geom = gdf_shores.dissolve(by="diss").reset_index()
        dissolved = pd.concat([dissolved, geom], ignore_index=True)
        dissolved.drop("diss", axis=1, inplace=True)

    return dissolved


def check_overlay(line_geometry, img_path):
    """Evaluates whether a line intesects the extent of a raster.
        Returns True if a valid intersection is found or False if not. In case of MultiLine features,
        evaluate if any of the lines intersects with the raster extent,
        which confirms that the CRS of both shapes geometries are correctly matched.

    Args:
        line_geometry (Shapely Line or MultiLinestring objects): geometry of line to evaluate its overlay on raster.
        img_path (str): Path to the geotiff to evaluate line overlay with.

    Returns:
        bool (bool): True, if a valid match is found. False, if the line do not intersect the raster."""

    # create a polygon with raster bounds
    with ras.open(img_path, "r") as dataset:

        ul = dataset.xy(0, 0)
        ur = dataset.xy(0, dataset.shape[1])
        lr = dataset.xy(dataset.shape[0], dataset.shape[1])
        ll = dataset.xy(dataset.shape[0], 0)

        ext_poly = gpd.GeoSeries(Polygon([ul, ur, lr, ll, ul]), crs=dataset.crs)

    # get geom_type
    if isinstance(line_geometry.geom_type, str):
        geom_type = line_geometry.geom_type
    elif isinstance(line_geometry.geom_type, pd.Series):
        geom_type = geom.geom_type[0]

    if geom_type == "MultiLineString":

        geom_in = list(line_geometry)[0]

        if ext_poly[0].intersects(geom_in):
            return True
        else:
            return False

    elif geom_type == "LineString":

        geom_in = line_geometry

        if ext_poly[0].intersects(geom_in):
            return True
        else:
            return False
    else:
        raise IOError("Shape is not a line.")

    # http://wikicode.wikidot.com/get-angle-of-line-between-two-points
    # angle between two points




def correct_multi_detections(shore_pts, transects):
    """Corrects for multiple points detections on shorelines transects. Multi detections occur when the shoreline bends considerably and the transects intersect it more than once.

    Args:
        shore_pts (np.array): array to be transformed.
        transects (pd.DataFRame): transects to correct.
    Returns:
        geometries (list): List of geometries of correct points.
    """

    geometries = []
    for i in range(shore_pts.shape[0]):
        pt_i = shore_pts.iloc[[i]]
        if pt_i.qa.values[0] == 0:
            geometries.append(np.nan)
        else:
            if pt_i.qa.values[0] != 1:

                start_line = transects.query(
                    f"tr_id=='{pt_i.tr_id.values[0]}'"
                ).geometry.boundary.iloc[0][0]

                min_idx = np.argmin(
                    [start_line.distance(pt) for pt in pt_i.geometry.values[0]]
                )
                geometries.append(pt_i.geometry.values[0][min_idx])

            else:
                geometries.append(pt_i.geometry.values[0])

    return geometries


def extract_shore_pts(
    transects,
    shore,
    crs=32754,
    tr_id_field="tr_id",
    date_field="date",
    raw_date_format="%Y%m%d",
):

    points = []
    tr_ids = []
    survey_dates = []

    for i in range(transects.shape[0]):
        transect_i = transects.iloc[i]
        try:
            point1 = transect_i.geometry.intersection(shore.unary_union)
        except BaseException:
            try:
                point1 = transect_i.geometry.intersection(shore)
            except BaseException:
                point1 = np.nan

        points.append(point1)
        tr_ids.append(transect_i.loc[tr_id_field])
        survey_dates.append(shore[date_field].values[0])

    df = pd.DataFrame({"geometry": points, "tr_id": tr_ids, "raw_date": survey_dates})

    replace_nan = [
        df.iloc[i].geometry if df.iloc[i].geometry.is_empty == False else np.nan
        for i in range(df.shape[0])
    ]
    qa = [
        0 if isinstance(i, float) else len(i) if i.geom_type == "MultiPoint" else 1
        for i in replace_nan
    ]

    df["geometry"] = replace_nan
    df["qa"] = qa
    df["geometry"] = correct_multi_detections(df, transects)

    gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=crs)
    #     gdf['date_formatted']=[datetime.datetime.strptime(str(raw_date),raw_date_format) for raw_date in gdf.raw_date]

    return gdf


def shore_shift(transects, gt, sat, crs, baseline_pts, sat_from_baseline=False):

    sat_pts = extract_shore_pts(transects, sat, crs)
    gt_pts = extract_shore_pts(transects, gt, crs, date_field="raw_date")

    sat_dists = baseline_pts.distance(sat_pts)
    gt_dists = baseline_pts.distance(gt_pts)

    baseline_pts["sat_dist"] = baseline_pts.distance(sat_pts)
    baseline_pts["gt_dist"] = baseline_pts.distance(gt_pts)
    baseline_pts.dropna(subset=["sat_dist", "gt_dist"], thresh=2, inplace=True)
    rmse = sqrt(mean_squared_error(baseline_pts.gt_dist, baseline_pts.sat_dist))

    shore_shift = sat_dists - gt_dists  # if negative, seaward bias.

    transects["rmse"] = rmse
    transects["std"] = shore_shift.std()
    transects["shore_shift"] = shore_shift

    if bool(sat_from_baseline):
        transects["sat_from_baseline"] = baseline_pts.distance(sat_pts)
    else:
        pass

    return transects


def rawdate_from_timestamp_str(timestamp_str):

    split = timestamp_str.split(" ")[0].split("-")
    return split[0] + split[2] + split[1]


def corr_baseline_distance(dist, slope, z_tide):
    """Applies a simple geometric tidal correction of the points along a shoreline

    Args:
        dist (float, pd.Series): Uncorrected distance or pd.Series of distances from the baseline.
        slope (float): Subaerial beach profile slope in percentage.
        z_tide (float): Water level or tidal height.

    Returns:
        float, list: The corrected value or list of corrected values.
    """

    if isinstance(dist, (int, float, complex)):

        tan_b = tan(radians(slope))
        return dist + (z_tide / tan_b)

    elif isinstance(dist, (pd.Series)):

        tans_b = [tan(radians(i)) for i in slope]
        corrs = [d + (z / sl) for d, sl, z in zip(dist, tans_b, z_tide)]

        return corrs
    else:
        raise TypeError("Input must be either Pandas.Series or numeric (int,float).")


def error_from_gt(
    shorelines,
    groundtruths,
    crs_dict_string,
    location,
    sampling_step,
    tick_length,
    shore_geometry_field,
    gt_geometry_field,
    gt_date_field="raw_date",
    shore_date_field="raw_date",
    side="both",
    baseline_mode="dynamic",
    tidal_correct=None,
):
    """Compute shorelines errors from a groundtruth references. You can use a fixed baseline shoreline or let the baseline be dynamic, which means
    that a new set of transects will be created on each groundtruth shoreline.

    Args:
        shorelines (gpd.GeoDataFrame): GeoDataFrame of shorelines to test.
        groundtruths (gpd.GeoDataFrame): GeoDataFrame of shorelines to test.
        crs_dict_string (dict): Dictionary storing location codes as key and crs information as values, in dictionary form.
        location (str): Strings of location code ("apo").
        sampling_step (int, float): Alongshore distanace to separate each evaluation transect.
        tick_length (int, float): Length of transects.
        shore_geometry_field (str): Field where the geometry of the test shoreline is stored.
        gt_geometry_field (str): Field where the geometry of the groundtruth shoreline is stored.
        gt_date_field (str): Field where the survey dates are stored in the groundtruth dataset.
        shore_date_field (str): Field where the survey dates are stored in the to-correct dataset.
        side (str): Whether to create transect on the right, left or both sides. Default to "both".
        baseline_mode (str): If "dynamic" (default), statistics will be computed from transects created from each groundtruth shoreline. If path to a .gpkg is provided, then use those arbitrary location specific baselines and transects will be fixed.
        tidal_correct (str, None): If str, apply tidal correction.
    Returns:
        pd.DataFrame: Dataframe containing the distances from groundtruth at each timestep.
    """
    crs = crs_dict_string[location]

    if os.path.isfile(baseline_mode):
        print("Fixed baseline mode selected.")
        baseline_loc = gpd.read_file(baseline_mode)

        if baseline_loc.crs != crs:
            baseline_loc = baseline_loc.to_crs(crs)
        else:
            pass

        transects = create_transects(
            baseline_loc,
            sampling_step=sampling_step,
            tick_length=tick_length,
            crs=crs,
            location=location,
        )
        df_temp = pd.DataFrame.from_dict(
            {
                "geometry": [
                    Point(tr_geom.coords[0][0], tr_geom.coords[0][1])
                    for tr_geom in transects.geometry
                ],
                "tr_id": transects.tr_id,
            }
        )
        baseline_pts = gpd.GeoDataFrame(df_temp, geometry="geometry", crs=crs)

    elif baseline_mode == "dynamic":
        print("Dynamic baseline mode selected.")

    else:
        raise TypeError(
            "Baseline mode must be either 'dynamic' or a valid path to a .gpkg."
        )

    shore_shift_df = pd.DataFrame()

    # subset CS shorelines with a location
    cs_shore_in = groundtruths.query(f"location=='{location}'")
    # select all Sentinel-2 shorelines in that location
    tests = shorelines.query(f"location=='{location}'")

    # if cs_shore_in.crs != crs:
    #     cs_shore_in=cs_shore_in.to_crs(crs)
    # else:
    #     pass
    #
    # if tests.crs != crs:
    #     tests=tests.to_crs(crs)
    # else:
    #     pass

    for i in range(cs_shore_in.shape[0]):  # for all CS groundtruths in location

        groundtruth = cs_shore_in.iloc[[i]]  # select the current groundtruth

        # get survey date           # HARDCODED DATE FIELD! BAD
        survey_date = groundtruth.loc[gt_date_field][0]

        if survey_date in tests.raw_date.unique():
            print(f"Working on {survey_date}...")

        else:
            print(
                f"Groundtruth in date {survey_date} not matched with any shorelines date."
            )

        if baseline_mode == "dynamic":
            # create transects and baselines pts dynamically from each groundtruth
            # shoreline
            transects = create_transects(
                groundtruth,
                sampling_step=sampling_step,
                tick_length=tick_length,
                crs=crs,
                location=location,
            )

            # create a geodataframe of transects starting points to compute distance
            # from
            df = pd.DataFrame.from_dict(
                {
                    "geometry": [
                        Point(tr_geom.coords[0][0], tr_geom.coords[0][1])
                        for tr_geom in transects.geometry
                    ],
                    "tr_id": transects.tr_id,
                }
            )
            baseline_pts = gpd.GeoDataFrame(df, geometry="geometry", crs=crs)

            # extract groundtruth distance from baseline (should be half transect,
            # i.e. transect centroid)
            gt_pts = extract_shore_pts(transects, groundtruth)

        transects.loc[:, "raw_date"] = survey_date

        # list all the satellite shorelines that corresponds to the groundtruth
        sat_shores_in = tests.query(f"raw_date=='{survey_date}'")

        for j in range(sat_shores_in.shape[0]):

            shore_sat = sat_shores_in.iloc[[j]]

            new_transects = shore_shift(
                transects=transects,
                gt=groundtruth,
                sat=shore_sat,
                crs=crs,
                baseline_pts=baseline_pts,
                sat_from_baseline=True,
            )

            shore_sat.rename({"geometry": "geom_shoreline"}, axis=1, inplace=True)
            merged = pd.merge(shore_sat, new_transects, on="raw_date")
            merged.rename({"location_x": "location"}, axis=1, inplace=True)

            shore_shift_df = pd.concat([shore_shift_df, merged], ignore_index=True)

    return shore_shift_df


def toes_from_slopes(
    series, distance_field="distance", slope_field="slope", sigma=0, peak_height=30
):
    """Returns dune toe distance (from transect origin) by extracting peaks higher of a given value
    from a Gaussian filtered slope profile. It can return multiple candidates when multiple peaks are found.
    These will be used to clip beachfaces, defined as going from the swash line to dune toes.
    Args:
        series (pd.Series): Slope profile.
        distance_field (str): Field where the distance is stored. Default to "distance".
        slope_field (str):  Field where the slope is stored. Default to "slope".
        sigma (int): Number of standard deviations sued in the Gaussian smoothing filter.
        peak_height (int): Threshold to use to define a peak in the smoothed slope profile.

    Returns:
        Toe distances of the given slope profile.
    """
    sorted_series = series.sort_values([distance_field])

    gauss = gaussian_filter(sorted_series.loc[:, slope_field], sigma=sigma)
    peak = sig.find_peaks(gauss, height=peak_height)

    try:
        toe_distances = sorted_series.iloc[peak[0]][distance_field]

    except BaseException:
        toe_distances = np.nan

    return toe_distances


def toes_candidates(
    df,
    location_field="location",
    date_field="raw_date",
    tr_id_field="tr_id",
    distance_field="distance",
    slope_field="slope",
    sigma=0,
    peak_height=30,
):
    """Dataframe implementation of the toes_from_slope function.
    Returns dune toe distance (from transect origin) by extracting peaks higher of a given value
    from a Gaussian filtered slope profile. It can return multiple candidates when multiple peaks are found.
    These will be used to clip beachfaces, defined as going from the swash line to dune toes.
    Args:
        df (pd.DataFrame): Dataframe containing the slope profiles.
        location_field (str): Field where the location code is stored. Default to "location".
        date_field (str): Field where the survey date is stored. Default to "raw_date".
        tr_id_field (str): Field where the transect ID is stored. Default to "tr_id".
        distance_field (str): Field where the distance is stored. Default to "distance".
        slope_field (str):  Field where the slope is stored. Default to "slope".
        sigma (int): Number of standard deviations sued in the Gaussian smoothing filter.
        peak_height (int): Threshold to use to define a peak in the smoothed slope profile.

    Returns:
        df_formatted (pd.DataFrame): Candidate distances of the each slope profile."""

    apply_dict = {
        "distance_field": distance_field,
        "slope_field": slope_field,
        "sigma": sigma,
        "peak_height": peak_height,
    }

    dist_toe_ = df.groupby([location_field, date_field, tr_id_field]).apply(
        toes_from_slopes, **apply_dict
    )

    dist_toe_df = pd.DataFrame(pd.Series(dist_toe_, name="toe_distances"))

    df_formatted = pd.DataFrame(
        dist_toe_df.reset_index()
        .groupby([location_field, date_field, tr_id_field])["toe_distances"]
        .apply(np.array)
    )

    return df_formatted


def consecutive_ids(data, indices=True, limit=1):
    """Returns indices of consecutive tr_ids in groups. Covenient to create multi-line geometries in case of disconnected shorelines."""

    if bool(indices) == False:
        return_i = 1  # return data values
    else:
        return_i = 0

    groups = []

    for k, g in groupby(enumerate(data), lambda ix: ix[0] - ix[1]):
        groups.append(list(map(itemgetter(return_i), g)))

    groups = [i for i in groups if len(i) > limit]

    return groups


def tidal_correction(
    shoreline,
    cs_shores,
    gdf,
    baseline_folder,
    crs_dict_string,
    limit_correction,
    mode,
    alongshore_resolution,
    slope_value="median",
    side="both",
    tick_length=200,
    subset_loc=None,
    limit_vertex=1,
    baseline_threshold="infer",
    replace_slope_outliers=True,
    save_trs_details=False,
    trs_details_folder=None,
    gdf_date_field="raw_date",
    distance_field="distance",
    date_field="raw_date",  # of transect geodataframe
    toe_field="toe",
):
    """
    Simple tidal correction for input shorelines. It can automatically extract subaerial beachfaces and more.

    Args:
        shoreline (gpd.GeoDataFrame): The geodataframe storing points along a shoreline.
        cs_shores (gpd.GeoDataFrame): The width and height of each single tile of the grid, given in the CRS unit (use projected CRS).
        gdf (gpd.GeoDataFrame): Coordinate Reference System in the dictionary format (example: {'init' :'epsg:4326'})
        baseline_folder (str): Path to the folder storing the baseline Geopackages (.gpkgs).
        crs_dict_string (dict): Dictionary storing location codes as key and crs information as values, in dictionary form.
        limit_correction (bool): If True, only use beachface slopes to compute the statistic for tidal correction. When False, retain the full transects to compute the slope statistics to correct shorelines. When a slope value is provided, it automatically sets to False. Defaults to True.
        mode (str): If 'sat', use satellite shorelines as seaward edge to classify beachfaces. If 'gt', use groundthruth shorelines instead.
        alongshore_resolution (str, float): The alongshore spacing between transects, in the unit of measure of the location CRS. If 'infer', use the gdf file to detect the spacing with 10cm precision. If the transects spacing is less than 10cm, set the spacing manually. Note: It also smoothes the original line if this value is greater of the original line vertex spacing.
        slope_value (int,float,'mean','median','min','max'): If a numeric value is provided (assumed to be in degrees), use it to correct the shorelines. If one of 'mean','median','min','max', use this statistics instead. It also computes range, standard deviation and variance for analytical purposes, despite should not be used to correct shorelines.
        side (str, 'left', 'right', 'both'): Whether if retain only the left, right or both sides of the transects once created. Defaults to 'both'.
        tick_length (int, float): Across-shore length of each transect in the unit of measure of the location CRS.
        subset_loc (list). List of string of location codes to limit the correction. Default to None.
        limit_vertex (int): Sets the minimum number of consecutive transect ids to create one segment of the corrected shoreline. Defaults to 1.
        baseline_threshold ("infer", float, None): If a number is provided, it sets the minimum distance the original un-corrected and the tidal-corrected shorelines must be in order to consider the correction valid. If the distance between the original the tidal-corrected shorelines at one point is less than this value, then the original shoreline is retained. If it is above, then the tidal-corrected value is retained. This is done to avoid to correct areas where an artificial shoreline   occurs (seawalls or harbours). If "infer", then the Otsu method is used to find this threshold value. This option works where artificial shorelines are present. If None, do not use this option. Default to "infer".
        replace_slope_outliers (bool): If True (default), replace the values of the defined slope statistics (slope_value parameter) with its survey-level median.
        save_trs_details (bool): True to save a .CSV file with transect-specific information. It defaults to False.
        trs_details_folder (str): Folder where to save the transect details. Defaults to None.
        gdf_date_field (str): Date field of the slope geodataframe used to correct the shoreline.Defaults to "raw_date".
        distance_field (str): Field storing the distance values. Default to "distance".
        date_field (str): Field storing the survey date values. Default to "raw_date".
        toe_field (str): Field storing the toe distances values. Default to "toe".

    Returns:
        corr_shorelines (gpd.GeoDataFrame): GeoDataFrame containing both the original shorelines and the corrected ones, stored in two different geometry fields.
    """

    if isinstance(slope_value, (int, float)):
        limit_correction = False
        extract_slope = False
        print(
            f"User-defined slope value of {slope_value} degrees will be used to correct shorelines. No beachfaces extracted."
        )

    else:
        limit_correction = True
        extract_slope = True

    if limit_correction:
        limited = "limited"
    else:
        limited = "notlimited"

    if "water_index" in shoreline.columns:
        dataset_type = "uavs_hores"
    else:
        dataset_type = "satellite_shores"

    grouping_fields = set(["water_index", "thr_type", "raw_date", "location"])
    cols = set(shoreline.columns.values)

    group_by_fields = list(cols.intersection(grouping_fields))

    # __________________ TIDAL CORRECTION_____________________________________

    big_final = pd.DataFrame()

    list_locations = gdf.location.unique()  # list the locations in the GDF file

    for location in list_locations:  # work iteratively on each location
        print(f"working on {location}")

        if isinstance(alongshore_resolution, (int, float)):
            print(
                f"Transect spacing set manually (alongshore resolution) = {alongshore_resolution} ."
            )
        elif alongshore_resolution == "infer":

            alongshore_resolution = infer_along_trs_spacing(gdf)
            print(
                f"Transect spacing (alongshore resolution) inferred = {alongshore_resolution} ."
            )
        else:
            raise NameError(
                "Alongshore resolution must be either a float, int, or 'infer'."
            )

        # get dates of location
        list_dates = gdf.query(f"location=='{location}'").raw_date.unique()
        shores_to_corr = shoreline.query(
            f"location=='{location}' & raw_date in @list_dates"
        )  # get shores to correct
        gt_shores = cs_shores.query(
            f"location=='{location}' & raw_date in @list_dates"
        )  # get groudtruths shores
        crs = crs_dict_string[location]  # get crs of location

        if shores_to_corr.crs != crs:
            shores_to_corr = shores_to_corr.to_crs(crs)
        else:
            pass

        if gt_shores.crs != crs:
            gt_shores = gt_shores.to_crs(crs)
        else:
            pass

        # read baseline as a GeoDataFrame
        try:
            baseline_location_path = glob.glob(f"{baseline_folder}/{location}*.gpkg")[0]
        except BaseException:
            raise NameError("Baseline file not found.")
        baseline = gpd.read_file(baseline_location_path, crs=crs)

        # create transects  # same parameter as the slope extraction. TO DO: if
        # exists, use existing data.
        transects = create_transects(
            baseline=baseline,
            side=side,
            sampling_step=alongshore_resolution,
            tick_length=tick_length,
            location=location,
            crs=crs,
        )

        # 1)_____________________________________________CREATE THE BASELINE POINT

        # create a geodataframe of transects starting points to compute distance from

        df_tmp = pd.DataFrame.from_dict(
            {
                "geometry": [
                    Point(tr_geom.coords[0][0], tr_geom.coords[0][1])
                    for tr_geom in transects.geometry
                ],
                "tr_id": transects.tr_id,
                "location": location,
            }
        )
        baseline_pts = gpd.GeoDataFrame(df_tmp, geometry="geometry", crs=crs)

        # loop through all transects in each satellite shoreline to compute
        # distances from the baseline points
        print("Computing cross-shore distances of satellite shorelines.")

        orig_shore_base_distances = pd.DataFrame()

        for i in tqdm(range(shores_to_corr.shape[0])):
            # if "water_index" column is present, we will need to group based on water
            # indices and threshold types too

            original_shore = shores_to_corr.iloc[[i]][[*group_by_fields, "geometry"]]

            original_shore_pts = extract_shore_pts(
                transects, original_shore, date_field="raw_date", crs=crs
            )

            matched = pd.merge(
                original_shore_pts,
                baseline_pts,
                how="left",
                on="tr_id",
                suffixes=("_pt_on_shore", "_pt_on_base"),
            )
            dists = []
            for shore_pt, base_pt in zip(
                matched.geometry_pt_on_shore, matched.geometry_pt_on_base
            ):
                try:
                    dists.append(shore_pt.distance(base_pt))
                except BaseException:
                    dists.append(np.nan)
            matched["sat_from_baseline"] = dists
            matched["raw_date"] = original_shore.raw_date.values[0]

            if "water_index" in shoreline.columns:

                matched["water_index"] = original_shore.water_index.values[0]
                matched["thr_type"] = original_shore.thr_type.values[0]
            else:
                pass

            orig_shore_base_distances = pd.concat(
                [
                    matched.dropna(subset=["sat_from_baseline"]),
                    orig_shore_base_distances,
                ],
                ignore_index=True,
            )

        # update transects with the baseline shore distances info
        orig_shore_base_distances = pd.merge(
            orig_shore_base_distances,
            transects[["tr_id", "geometry"]],
            on="tr_id",
            how="left",
        )
        shores_to_corr.raw_date.astype(int)

        # loop through all transects in each groundthruth shoreline to compute
        # distances from the baseline points
        print("Computing cross-shore distances of UAV shorelines.")

        if mode == "gt":  # we will use these points to limit the beachface extraction

            gt_base_distances = pd.DataFrame()

            for shoreline_i in tqdm(range(gt_shores.shape[0])):
                shore_i = gt_shores.iloc[[shoreline_i]][
                    ["location", "raw_date", "geometry"]
                ]
                shore_pts_gt = extract_shore_pts(
                    transects, shore_i, date_field="raw_date"
                )

                matched = pd.merge(
                    shore_pts_gt,
                    baseline_pts,
                    how="left",
                    on="tr_id",
                    suffixes=("_gt_on_shore", "_gt_on_base"),
                )
                dists = []
                for shore_pt, base_pt in zip(
                    matched.geometry_gt_on_shore, matched.geometry_gt_on_base
                ):
                    try:
                        dists.append(shore_pt.distance(base_pt))
                    except BaseException:
                        dists.append(np.nan)
                matched["gt_from_baseline"] = dists
                matched["raw_date"] = shore_i.raw_date.values[0]

                gt_base_distances = pd.concat(
                    [matched.dropna(subset=["gt_from_baseline"]), gt_base_distances],
                    ignore_index=True,
                )

            gt_base_distances = pd.merge(
                gt_base_distances,
                transects[["tr_id", "geometry"]],
                on="tr_id",
                how="left",
            )
        else:
            pass

        # _______ EXTRACT BEACHFACE BASED ON SHORELINE POSITION AND DUNE TOE______

        if bool(limit_correction):

            print(f"Extracting dune toes from slope profiles.")
            gdf_loc = gdf.query(f"location=='{location}'")

            toes_cand = toes_candidates(gdf_loc, date_field="raw_date")

            if mode == "gt":

                baseline_distances_in = gt_base_distances
                baseline_field = "gt_from_baseline"
                txt = "UAV-derived"

                base_dist_toes = pd.merge(
                    baseline_distances_in,
                    toes_cand.reset_index(),
                    on=["location", "raw_date", "tr_id"],
                    how="left",
                )

            else:

                baseline_distances_in = orig_shore_base_distances
                baseline_field = "sat_from_baseline"
                txt = "satellite-derived"

                base_dist_toes = pd.merge(
                    baseline_distances_in,
                    toes_cand.reset_index(),
                    on=["location", "raw_date", "tr_id"],
                    how="left",
                )

            toes = []
            for t, d in zip(
                base_dist_toes.loc[:, "toe_distances"],
                base_dist_toes.loc[:, baseline_field],
            ):
                try:
                    toes.append(t[np.where(t <= d)][-1])
                except BaseException:
                    toes.append(np.nan)

            base_dist_toes["toe"] = toes

            print(
                f"Classifying beachfaces as going from {txt} shorelines to dune toes."
            )

            # preprocessing
            if mode == "gt":
                tmp_lookup = (
                    base_dist_toes.groupby(["location", "raw_date", "tr_id"])[
                        toe_field, baseline_field
                    ]
                    .first()
                    .reset_index()
                )
                tmp_merge = pd.merge(
                    tmp_lookup,
                    gdf_loc,
                    how="right",
                    on=["location", "raw_date", "tr_id"],
                )
                tmp_merge = tmp_merge.dropna(subset=[baseline_field])[
                    [
                        "distance",
                        baseline_field,
                        "location",
                        "raw_date",
                        "tr_id",
                        "toe",
                        "slope",
                    ]
                ]
            else:

                tmp_lookup = (
                    base_dist_toes.groupby([*group_by_fields, "tr_id"])[
                        toe_field, baseline_field
                    ]
                    .first()
                    .reset_index()
                )
                tmp_merge = pd.merge(
                    tmp_lookup,
                    gdf_loc,
                    how="right",
                    on=["location", "raw_date", "tr_id"],
                )
                tmp_merge = tmp_merge.dropna(subset=[baseline_field])[
                    [
                        "distance",
                        baseline_field,
                        "location",
                        "raw_date",
                        "tr_id",
                        "toe",
                        "slope",
                    ]
                ]

            # conditions
            tmp_merge["beachface"] = [
                "bf"
                if tmp_merge.loc[i, distance_field] >= tmp_merge.loc[i, toe_field]
                and tmp_merge.loc[i, distance_field] <= tmp_merge.loc[i, baseline_field]
                else "land"
                if tmp_merge.loc[i, distance_field] < tmp_merge.loc[i, toe_field]
                else "water"
                if tmp_merge.loc[i, distance_field] > tmp_merge.loc[i, toe_field]
                else np.nan
                for i in range(tmp_merge.shape[0])
            ]

        else:  # if we do not want to limit the correction to the beachface statistics, but we want to use the full transect stats

            merged_tmp = pd.merge(
                shores_to_corr,
                orig_shore_base_distances,
                on=[*group_by_fields],
                how="right",
                suffixes=("_shore", "_tr"),
            )

        if isinstance(slope_value, str):

            if bool(limit_correction):

                df_in = tmp_merge.query("beachface=='bf'")
            else:

                df_in = pd.merge(
                    merged_tmp.astype({"tr_id": int, "raw_date": int, "location": str}),
                    gdf_loc.astype({"tr_id": int, "raw_date": int, "location": str})[
                        ["location", "raw_date", "tr_id", "slope"]
                    ],
                    how="right",
                    on=["location", "raw_date", "tr_id"],
                )

            ops_dict = {
                "mean": np.nanmean,
                "median": np.nanmedian,
                "min": np.min,
                "max": np.max,
                "range": np.ptp,
                "std": np.nanstd,
                "var": np.nanvar,
            }

            stats = pd.DataFrame()
            for i in ops_dict.keys():

                tmp = pd.Series(
                    df_in.groupby(["raw_date", "tr_id"]).slope.apply(ops_dict[i]),
                    name=f"{i}_beachface_slope",
                )
                stats = pd.concat([stats, tmp], axis=1)
                stats = stats.set_index(
                    pd.MultiIndex.from_tuples(stats.index, names=("raw_date", "tr_id"))
                )
            stats["location"] = location

            slope_field = [
                stat for stat in stats.columns if stat.startswith(slope_value)
            ][0]

            if bool(limit_correction):
                print(f"Using {slope_field} of beachfaces to correct shorelines.")
            else:
                print(f"Using full transect {slope_field} to correct shorelines.")

            stats = stats.reset_index()
            stats["raw_date"] = [int(i) for i in stats.raw_date]

            orig_shore_trs_stats = pd.merge(
                orig_shore_base_distances,
                stats,
                on=["tr_id", "location", "raw_date"],
                how="left",
            )

            if bool(replace_slope_outliers):

                # create temp dataframes to store survey-level slope values stats and
                # derive 3 sigmas thresholds per date
                survey_slope_field_stat = (
                    stats.groupby(["raw_date"])[slope_field].describe().reset_index()
                )
                survey_slope_field_stat["sigmas_3"] = (
                    survey_slope_field_stat.loc[:, "50%"] * 3
                )

                orig_shore_trs_stats = pd.merge(
                    orig_shore_trs_stats,
                    survey_slope_field_stat[["raw_date", "sigmas_3", "50%"]],
                    on=["raw_date"],
                    how="left",
                )

                # add the field 'outlier' and mark slope_values exceeding 3 std.
                orig_shore_trs_stats.loc[:, "outlier"] = np.where(
                    orig_shore_trs_stats[slope_field]
                    > orig_shore_trs_stats["sigmas_3"],
                    True,
                    False,
                )

                # replace outliers with median survey values.
                orig_shore_trs_stats.loc[:, slope_field] = np.where(
                    orig_shore_trs_stats[slope_field]
                    > orig_shore_trs_stats["sigmas_3"],
                    orig_shore_trs_stats["50%"],
                    orig_shore_trs_stats[slope_field],
                )
            else:
                pass

            correction_series = orig_shore_trs_stats.loc[:, slope_field]

        elif isinstance(slope_value, (int, float)):
            print(f"Using user-provided slope of {slope_value} to correct shorelines.")
            orig_shore_trs_stats = orig_shore_base_distances.assign(
                fixed_slope=slope_value
            )

            correction_series = orig_shore_trs_stats.fixed_slope

        # add tide height

        orig_shore_trs_stat_tide = pd.merge(
            orig_shore_trs_stats,
            shores_to_corr[[*group_by_fields, "tide_height"]],
            on=[*group_by_fields],
            how="left",
        )

        orig_shore_trs_stat_tide["corr_dist"] = corr_baseline_distance(
            orig_shore_trs_stat_tide.sat_from_baseline,
            correction_series,
            orig_shore_trs_stat_tide.tide_height,
        )

        orig_shore_trs_stat_tide["diff_corr"] = (
            orig_shore_trs_stat_tide.sat_from_baseline
            - orig_shore_trs_stat_tide.corr_dist
        )

        orig_shore_corr_dist_gdf = gpd.GeoDataFrame(
            orig_shore_trs_stat_tide, geometry="geometry", crs=crs
        )

        # _______ Apply threshold to correction, below this threshold, it is not c

        diffs = orig_shore_corr_dist_gdf.diff_corr.values
        corrs = orig_shore_corr_dist_gdf.corr_dist.values
        originals = orig_shore_corr_dist_gdf.sat_from_baseline.values

        # whether to correct or not shorelines closer to this threshold
        if baseline_threshold == "infer":
            baseline_threshold_value = np.round(
                threshold_multiotsu(
                    orig_shore_corr_dist_gdf.diff_corr.dropna(), classes=2
                )[0],
                2,
            )
            print(f"Inferred baseline_threshold of: {baseline_threshold_value} meters")

        elif isinstance(baseline_threshold, (int, float)):
            baseline_threshold_value = baseline_threshold
            print(f" Baseline distance threshold of: {baseline_threshold_value} meters")

        elif baseline_threshold is None:
            pass
        else:
            raise ValueError(
                "Baseline_threshold must be either 'infer' or a numeric value."
            )

        updated_corr_dists = [
            c if d > baseline_threshold_value else o
            for d, c, o in zip(diffs, corrs, originals)
        ]
        orig_shore_corr_dist_gdf["updated_corr_dists"] = updated_corr_dists

        if "slope_field" in locals():
            a = orig_shore_corr_dist_gdf.dropna(subset=[slope_field])[
                [
                    *group_by_fields,
                    "tr_id",
                    "sat_from_baseline",
                    "updated_corr_dists",
                    "geometry",
                ]
            ]
        else:
            a = orig_shore_corr_dist_gdf.dropna()[
                [
                    *group_by_fields,
                    "tr_id",
                    "sat_from_baseline",
                    "updated_corr_dists",
                    "geometry",
                ]
            ]
        a["updated_corr_dists"] = [
            a.iloc[i].sat_from_baseline
            if a.iloc[i].updated_corr_dists < 0
            else a.iloc[i].updated_corr_dists
            for i in range(a.shape[0])
        ]
        b = a.groupby([*group_by_fields])["tr_id"].apply(np.array).reset_index()
        b["updated_corr_dists"] = (
            a.groupby([*group_by_fields])["updated_corr_dists"]
            .apply(np.array)
            .reset_index()["updated_corr_dists"]
        )
        b["tr_geometries"] = (
            a.groupby([*group_by_fields])["geometry"]
            .apply(np.array)
            .reset_index()["geometry"]
        )

        # 5)_________________ Lines creation_________________________________________

        lines = []

        for i, row in b.iterrows():
            groups = consecutive_ids(data=row.tr_id, limit=limit_vertex)

            if len(groups) > 0:

                lines_group = []
                for j, group in enumerate(groups):

                    trs = row.tr_geometries[group]
                    dis = row.updated_corr_dists[group]

                    line_pts = []
                    for distance, transect in zip(dis, trs):
                        point = transect.interpolate(distance)
                        line_pts.append(point)

                    line = LineString(line_pts)
                    lines_group.append(line)

                if len(lines_group) > 1:
                    lines_row = unary_union(lines_group)
                else:
                    lines_row = lines_group[0]

                lines.append(lines_row)
            else:
                print(
                    f"WARNING:\n\
                {row.location}_{row.raw_date}_{row.water_index}_{row.thr_type} has {len(groups)} groups. Skipped."
                )
                lines.append(np.nan)

        lines = gpd.geoseries.GeoSeries(lines, crs=crs)
        b["corr_shore_geometry"] = lines.geometry
        final = pd.merge(shores_to_corr, b, on=[*group_by_fields], how="left")

        # 6)_________________ saving outputs________________________________________

        if bool(save_trs_details):
            trs_details_file_name = (
                f"trsdetails_{location}_{slope_value}_{limited}_{mode}.csv"
            )
            trs_details_out_path = os.path.join(
                trs_details_folder, trs_details_file_name
            )

            orig_shore_corr_dist_gdf.to_csv(trs_details_out_path, index=False)
            print(f"File {trs_details_file_name} saving in {trs_details_folder}.")
        else:
            pass

        big_final = pd.concat([big_final, final], ignore_index=True)

        print(f"Done with {location}.")

    return big_final


def save_slope_corr_files(
    dsm_folder,
    baseline_folder,
    slope_profiles_folder,
    loc_codes,
    crs_dict_string,
    shoreline,
    across_shore_resolution,
    alongshore_resolution,
    tick_length,
    side,
    transects_gdf=None,
    exclude_loc=None,
    only_loc=None,
):

    list_locations = list(
        set([getLoc(i, loc_codes) for i in glob.glob(f"{dsm_folder}/*.tif*")])
    )  # get unique locations from list of DSMs

    if only_loc is not None and exclude_loc is not None:
        raise ValueError(
            "Both only_loc and exclude_loc have been provided.\nSet both to None to work on all locations (default) or provide only one parameter to exclude or only work on specified locations."
        )
    else:
        pass

    if only_loc is not None:
        list_locations = (
            only_loc  # list_locations is only those specified in only_loc parameter
        )
        print(f"Working only on locations: {only_loc}")

    elif exclude_loc is not None:
        list_locations = [loc for loc in list_locations if loc not in exclude_loc]
        print(f"Excluded locations: {exclude_loc}")

        print("Locations to extract slope profiles from:\n")
        for loc in list_locations:
            print(f"{loc}")

    else:
        pass

    list_dsm = glob.glob(f"{dsm_folder}/*.tif*")

    # _______________Output Files preparations________________________________________

    if isinstance(across_shore_resolution, float):
        across_shore_resolution_txt = str(across_shore_resolution).replace(".", "dot")
    else:
        across_shore_resolution_txt = str(across_shore_resolution)

        # Check if baselines CRS match rasters CRSs

    inconsistencies = pd.DataFrame()

    for location in list_locations:

        baseline = gpd.read_file(
            glob.glob(f"{baseline_folder}\\{location}*")[0], geometry="geometry"
        )

        imgs = glob.glob(f"{dsm_folder}\\*{location}*")

        base_crs = baseline.crs["init"].split(":")[-1]
        raster_crs = [
            True if str(getCrs_from_raster_path(imgs[i])) != base_crs else False
            for i in range(len(imgs))
        ]

        inconsis = [img for incons, img in zip(raster_crs, imgs) if incons]
        if len(inconsis) != 0:

            df_temp = pd.DataFrame(
                {
                    "location": location,
                    "inconsis": inconsis,
                    "raster_crs": [
                        getCrs_from_raster_path(imgs[i]) for i in range(len(imgs))
                    ],
                    "baseline_crs": base_crs,
                }
            )
            inconsistencies = pd.concat([df_temp, inconsistencies], ignore_index=True)

    if inconsistencies.shape[0] != 0:
        raise ValueError(
            f"CRS error. These rasters have different CRSs from their baseline CRS. Please reproject the baseline to match DSMs."
        )
        return inconsistencies
    else:
        pass

    # _______________SLOPE COMPUTATION________________________________________

    print(
        "Preparing to compute slope rasters of all dates in the shorelines to correct.\n"
    )
    for location in list_locations:

        gdf = pd.DataFrame()

        print(f"Working on {location} .")
        slope_file_name = f"slopeprofiles_{location}_{alongshore_resolution}_{tick_length}_{across_shore_resolution_txt}.csv"

        crs = crs_dict_string[location]  # get crs of location

        baseline_location_path = [
            os.path.join(baseline_folder, baseline)
            for baseline in os.listdir(baseline_folder)
            if location in baseline
        ][0]
        baseline = gpd.read_file(baseline_location_path, crs=crs)

        # get raw_dates of location
        list_dates = list(shoreline.query(f"location=='{location}'").raw_date.unique())
        locs = []
        dates = []
        for i in list_dsm:
            locs.append(getLoc(i, list_loc_codes=loc_codes))
            dates.append(getDate(i))
        datasets = pd.DataFrame({"loc": locs, "date": dates})

        list_dates = [
            date_in
            for date_in in list_dates
            if date_in in datasets.query(f"loc=='{location}'").date.unique()
        ]

        # 0) Creating the extraction transects if not provided

        if len(list_dates) == 0:

            print(f"No DSMs found for {location}.")
            pass

        else:

            if isinstance(transects_gdf, gpd.GeoDataFrame):
                print(f"Transects GeoDataFrame provided.")
                transects = transects_gdf

            elif transects_gdf is None:
                transects = create_transects(
                    baseline=baseline,
                    side=side,
                    sampling_step=alongshore_resolution,
                    tick_length=tick_length,
                    location=location,
                    crs=crs,
                )

            else:
                raise ValueError(
                    "Transects parameter is either None (default) or a GeoDataFrame."
                )

            for raw_date in list_dates:

                transects["raw_date"] = raw_date

                # select the right dsm dataset
                dsm_path = [
                    dsm
                    for dsm in list_dsm
                    if str(raw_date) == str(getDate(dsm))
                    and location == getLoc(dsm, list_loc_codes=loc_codes)
                ]

                # double-check only one right DSM has been found
                if len(dsm_path) > 1:
                    raise ValueError(
                        f"Multiple datasets for location: {location} and date: {raw_date} have been found."
                    )
                elif len(dsm_path) == 0:
                    raise ValueError(
                        f"No datasets for location: {location} and date: {raw_date} have been found."
                    )
                else:
                    print(
                        f"Transects: location={location}; Date={raw_date} matched with DSM path= {dsm_path[0]}"
                    )

                # computing the slope raster
                print(
                    f"Loading raster DSM into memory, for {location} at date: {raw_date}."
                )
                terr = rd.LoadGDAL(dsm_path[0])
                print(f"Computing its slope in degrees.")
                slope = rd.TerrainAttribute(terr, attrib="slope_degrees")
                del terr  # to save memory
                print(f"Slope raster created.")

                # extracting slope

                tr_list = transects.tr_id.to_list()
                print(f"Extracting profiles.")
                for i in tr_list:
                    temp = get_profiles(
                        dsm=dsm_path[0],
                        transect_file=transects,
                        transect_index=i,
                        step=across_shore_resolution,
                        location=location,
                        date_string=raw_date,
                        add_xy=True,
                        add_terrain=slope,
                    )

                    temp.loc[:, "location"] = location
                    gdf = pd.concat([gdf, temp], ignore_index=True)

            # replacing DSM NoData (in Pix4D is - 10000.0) and 0.0 slopes with np.nan
            gdf["z"].replace(to_replace=-10000.0, value=np.nan, inplace=True)
            gdf["slope"].replace(
                to_replace=[0.0, -9999.0], value=[np.nan, np.nan], inplace=True
            )

            gdf["slope_tan_b"] = [tan(radians(slope_deg)) for slope_deg in gdf.slope]

            gdf["trs_along_space"] = alongshore_resolution
            gdf["trs_across_space"] = across_shore_resolution
            gdf["trs_length"] = tick_length
            gdf["side"] = side

            # Saving the data
            slope_profiles_out_path = os.path.join(
                slope_profiles_folder, slope_file_name
            )
            gdf.to_csv(slope_profiles_out_path, index=False)

        print(f"Correction profiles extraction finished and saved for {location}.\n")

        print("Ready to correct shorelines.")
    return gdf


def partial_tile_padding(
    dataset,
    geom,
    expected_shape,
    crs,
    tile_name,
    output_path,
    nodata,
    source_idx,
    height_idx,
    width_idx,
    out_idx,
    count,
    driver,
    geotransform,
):

    ## This will be the tile name

    ## Create a padded tile from the mask coordinates

    px_size = dataset.transform[0]
    ulx = geom.bounds[0]
    ury = geom.bounds[3]

    ulx_px_coords = floor(int(ulx) / 10) * 10
    ury_px_coords = ceil(int(ury) / 10) * 10

    pad_transform = from_origin(ulx_px_coords, ury_px_coords, px_size, px_size)

    pad_array = ras.features.rasterize(
        [mapping(geom)], out_shape=expected_shape, all_touched=False
    )
    print(f"pad_array:{pad_array.shape}")

    # pad_array_3=np.array([pad_array, pad_array, pad_array], ras.int16)
    pad_array_3 = (
        np.dstack([pad_array] * count)
        .astype(ras.int16)
        .reshape((count, expected_shape[0], expected_shape[1]))
    )

    print(f"pad_array_3:{pad_array_3.shape}")
    print(f"height_idx:{height_idx}")
    print(f"width_idx:{width_idx}")

    with MemoryFile(filename="padded") as memfile_padded:
        with memfile_padded.open(
            driver=driver,
            height=pad_array_3.shape[1],
            width=pad_array_3.shape[2],
            count=count,
            dtype=pad_array_3.dtype,
            crs=crs,
            transform=pad_transform,
        ) as mem_dataset_padded:

            mem_dataset_padded.write(pad_array_3, indexes=list(np.arange(1, count + 1)))

            # Create a the temporary in-memory partial tile

            out_image, out_transform = rasmask.mask(
                dataset=dataset,
                shapes=[mapping(geom)],
                invert=False,
                crop=True,
                filled=False,
                nodata=nodata,
                indexes=source_idx,
            )

            out_meta = dataset.meta
            print(f"out_image:{out_image.shape}")

            # format name: LocationCode_GridId_Date.tif
            if driver == "PNG":
                ext = "png"
            elif driver == "GTiff":
                ext = "tif"
            else:
                raise NameError("Driver must be either 'PNG' or 'GTiff'.")

            if driver == "PNG":
                ext = "png"
                out_meta.update({"dtype": ras.uint16})
            else:
                ext = "tif"

            out_meta.update(
                {
                    "height": out_image.shape[height_idx],
                    "width": out_image.shape[width_idx],
                    "transform": out_transform,
                    "count": count,
                }
            )

            with MemoryFile() as memfile:
                with memfile.open(**out_meta) as mem_dataset:

                    if driver == "PNG":
                        mem_dataset.write(out_image.astype(ras.uint16), indexes=out_idx)
                    else:
                        mem_dataset.write(out_image, indexes=out_idx)

                    # mosaic in-memory rasters and write to disk
                    two_rasters = [mem_dataset, mem_dataset_padded]

                    mosaic, out_trans = merge(
                        datasets=two_rasters,
                        indexes=list(np.arange(1, count + 1)),
                        nodata=nodata,
                        method="max",
                    )

                    print(f"mosaic:{mosaic.shape}")

                    savetxt = os.path.join(output_path, tile_name)

                    with ras.open(
                        savetxt,
                        "w",
                        driver=driver,
                        height=mosaic.shape[1],
                        width=mosaic.shape[2],
                        count=count,
                        dtype=mosaic.dtype,
                        crs=crs,
                        transform=out_trans,
                    ) as final_mosaic_tile:

                        final_mosaic_tile.write(mosaic)

                    print(f"Successfully saved partial tile in-memory: {tile_name} .")
                    if geotransform == True:

                        geot_series = pd.Series(
                            {"tile_code": f"{tile_code}", "geotransform": out_trans}
                        )

                        print("Tile geotransform returned.")
                        return geot_series


def tile_to_disk(
    dataset,
    geom,
    crs,
    tile_name,
    output_path,
    nodata,
    source_idx,
    height_idx,
    width_idx,
    out_idx,
    count,
    driver,
    geotransform,
):

    try:
        out_image, out_transform = rasmask.mask(
            dataset=dataset,
            shapes=[mapping(geom)],
            invert=False,
            crop=True,
            filled=False,
            nodata=nodata,
            indexes=source_idx,
        )
    except:
        out_image, out_transform = rasmask.mask(
            dataset=dataset,
            shapes=geom,
            invert=False,
            crop=True,
            filled=False,
            nodata=nodata,
            indexes=source_idx,
        )
    out_meta = dataset.meta

    if driver == "PNG":
        out_meta.update({"dtype": ras.uint16})

    out_meta.update(
        {
            "driver": driver,
            "height": out_image.shape[height_idx],
            "width": out_image.shape[width_idx],
            "transform": out_transform,
            "count": count,
            "crs": crs,
        }
    )

    savetxt = os.path.join(output_path, tile_name)

    with ras.open(savetxt, "w", **out_meta) as dest:
        if driver == "PNG":
            dest.write(out_image.astype(ras.uint16), indexes=out_idx)
        else:
            dest.write(out_image, indexes=out_idx)

    print(f"Successfully saved tile: {tile_name} .")
    if geotransform == True:

        geot_series = pd.Series(
            {"tile_code": f"{tile_code}", "geotransform": out_transform}
        )

        print("Tile geotransform returned.")
        return geot_series


def tiles_from_grid(
    grid,
    img_path,
    output_path,
    list_loc_codes,
    crs_dict_string,
    mode="rgb",
    sel_bands=None,
    driver="PNG",
    geotransform=False
):
    """
    Returns a dataframe with location, raw_date, filenames (paths) or geopackage index and CRS of each raster and its associated vector files.
    If the directory containing the vector files has only one file, it is assumed that this file stores vectors
    with location and raw_date columns.

    Args:
        grid (GeoDataFrame): GeoDataFrame of the grid of only tiles containing the line. Output of grid_from_shore function.
        img_path (str): Path of the directory containing the geotiffs datasets (.tiff or .tif).
        output_path (str): Path of the directory where to save the images tiles.
        list_loc_codes (list): list of strings containing location codes.
        mode (str,'rgb','mask','multi','custom'): 'rgb', the output images are 3-channels RGB tiles. 'mask', 1-band output tiles.
        'multi', multibands output tiles (with all input bands).
        'custom', use selected band indices (with sel_bands parameter) to only extract those bands from input multiband images
        (NOTE: in 'custom' mode, output tile bands indices are reindexd, so do not corresponds with the original band indices, but restart from 1).
        geotransform (bool or 'only'): If True, save tiles and also return a dictionary with the geotransform of each grid.
        If False, save tiles without geotransform dictionary. If "only", do not save tiles but return the geotransform dictionary only.
        sel_bands (list): list of integers (minimum is 1, not zero-indexed) corresponding to the bands to be used to create the tiles. Only used with mode='custom'.
        Default is None.
        driver (str, "GTiff" or "PNG"): tiles image file type. Default is "PNG".

    Returns:
        Saves tiles to the specified output folder and optionally return the tiles geotransform dictionary.
    """

    loc = getLoc(img_path, list_loc_codes)
    crs = crs_dict_string[loc]

    if driver == "PNG":
        ext = "png"
    elif driver == "GTiff":
        ext = "tif"
    else:
        raise NameError("Driver must be either 'PNG' or 'GTiff'.")

    with ras.open(img_path, "r") as dataset:

        if mode == "rgb":

            count = 3
            source_idx = [1, 2, 3]  # the band indices of the source image
            out_idx = source_idx
            sel_bands = None

            height_idx = 1
            width_idx = 2

        elif mode == "multi":

            if driver == "PNG":
                print(
                    "NOTE: PNG format doesn't support multibands. Returning GeoTiffs instead."
                )
                driver = "GTiff"
            else:
                pass

            sel_bands = None
            count = dataset.count
            source_idx = list(dataset.indexes)
            out_idx = None
            height_idx = 1
            width_idx = 2

        elif mode == "mask":

            count = 1
            source_idx = 1
            out_idx = 1
            height_idx = 0
            width_idx = 1

        elif mode == "custom":
            if len(sel_bands) > 3 and driver == "PNG":
                print(
                    "NOTE: More than 3 bands selected for the creation of PNG images. PNG format doesn't support multibands. Returning GeoTiffs instead."
                )
                driver = "GTiff"
            else:
                pass

            source_idx = sel_bands
            out_idx = None
            count = len(sel_bands)
            height_idx = 1
            width_idx = 2

        # creates gereferenced bounding box of the image
        geom_im = gpd.GeoSeries(box(*dataset.bounds), crs=crs_dict_string[loc])

        # evaluates which tiles are fully within the raster bounds
        fully_contains = [
            geom_im.geometry.contains(mask_geom)[0] for mask_geom in grid.geometry
        ]

        # get the expected shape of fully contained tiles
        full_in_geom = grid[fully_contains].iloc[[0]]["geometry"]
        geom_wdw = geometry_window(dataset, full_in_geom)
        expected_shape = (geom_wdw.height, geom_wdw.width)

        print(f"Expected shape{expected_shape}")

        for i, row in grid.iterrows():

            tile_name = f"{loc}_{row.grid_id}_{getDate(img_path)}_{mode}.{ext}"

            geom = Polygon(  # create the square polygons to clip the raster with
                (
                    (row.ulx, row.uly),
                    (row.urx, row.ury),
                    (row.lrx, row.lry),
                    (row.llx, row.lly),
                    (row.ulx, row.uly),
                )
            )

            # get the future shape of the tile which is about to get created
            geom_wdw = geometry_window(dataset, [mapping(geom)])
            tile_shape = (geom_wdw.height, geom_wdw.width)

            if tile_shape == expected_shape:

                tile_to_disk(
                    dataset=dataset,
                    geom=geom,
                    crs=crs,
                    tile_name=tile_name,
                    output_path=output_path,
                    nodata=0,
                    source_idx=source_idx,
                    height_idx=height_idx,
                    width_idx=width_idx,
                    out_idx=out_idx,
                    count=count,
                    driver=driver,
                    geotransform=geotransform,
                )

            else:

                partial_tile_padding(
                    dataset=dataset,
                    expected_shape=expected_shape,
                    crs=crs,
                    geom=geom,
                    tile_name=tile_name,
                    output_path=output_path,
                    nodata=0,
                    source_idx=source_idx,
                    height_idx=height_idx,
                    width_idx=width_idx,
                    out_idx=out_idx,
                    count=count,
                    driver=driver,
                    geotransform=geotransform,
                )


def arr2geotiff(
    array,
    transform,
    location,
    crs_dict_string,
    shape=(64, 64, 1),
    driver="GTiff",
    dtype=np.float32,
    save=None):
    """Transform an array into a Geotiff given its Shapely transform and location code.

    Args:
        array (array): array to be transformed.
        transform (tuple): tuple with Shapely transform parameters.
        location (str): Location code of the shoreline to convert.
        crs_dict_string (dict):  Dictionary storing location codes as key and crs information as values, in dictionary form.
        shape (tuple): Tuple of size 3 of the shape of the array. Default to (64,64,1)
        driver ("GTiff"): Driver used by Fiona to save file.
        dtype (data type object): Default to numpy.float32.
        save (None,path). If a full path is provided, save the file to a geotiff image (C:\my\new\image.tif). If None (default), the geotiff is saved in the memory but not to the disk.

    Returns:
        mem_dataset (rasterio.io.MemoryFile): Geotiff image saved into memory or optionally saved to a new raster.

    """

    with MemoryFile() as memfile:
        mem_dataset = memfile.open(
            driver="GTiff",
            height=array.shape[0],
            width=array.shape[1],
            count=array.shape[2],
            dtype=dtype,
            transform=transform,
            crs=crs_dict_string[location],
        )

        mem_dataset.write(array.reshape((shape[0], shape[1])), indexes=shape[2])

        if save != None:
            with ras.open(
                save,
                "w",
                driver=driver,
                height=array.shape[0],
                width=array.shape[1],
                count=array.shape[2],
                dtype=dtype,
                transform=transform,
                crs=crs_dict_string[location],
            ) as dest:
                if driver == "PNG":
                    dest.write(mem_dataset.astype(ras.uint16), indexes=1)
                else:
                    dest.write(array.reshape((shape[0], shape[1])), indexes=shape[2])

        return mem_dataset


def shoreline_from_prediction(
    prediction, z, geotransform, min_vertices=2, shape=(64, 64)
):
    """Obtain a georeferenced shoreline in a GeoDataFrame from a binary predicted water mask.
    Credits: adapted from Dr. Robbie Bishop-Taylor functions in Digital Earth Australia scripts, available at:
    https://github.com/GeoscienceAustralia/dea-notebooks/blob/develop/Scripts/dea_coastaltools.py

    Args:
        prediction (array): array to be thresholded.
        z (int, float): threshold value to separate land and water from the prediction array.
        geotransform (tuple): uple with Shapely transform parameters.
        min_vertices (int): Minimum number of vertices a segment has to have to be retained as a shoreline. Default to 2.
        shape (data type object): Shape of the array. Default to (64,64).
    Returns:
        Geodataframe with georeferenced shoreline.

    """

    # get shoreline
    shore_arr = contours_to_multiline(
        prediction.reshape(shape), z, min_vertices=min_vertices
    )

    # create geoseries and geodataframe
    shore_arr_geoseries = gpd.GeoSeries(shore_arr, name="geometry")
    contours_gdf = gpd.GeoDataFrame(shore_arr_geoseries, geometry="geometry")

    # georeference line using tile geotransform
    contours_gdf["geometry"] = contours_gdf.affine_transform(shapely_affine)

    return contours_gdf



def LISA_site_level(
    dh_df,
    crs_dict_string,
    mode,
    distance_value=None,
    decay=None,
    k_value=None,
    geometry_column='geometry'):
    """Performs Hot-Spot analysis using Local Moran's I as LISA for all the survey.
        Please refer to PySAL package documentation for more info.

    Args:
        dh_df (pd.DataFrame, str): Pandas dataframe or local path of the timeseries files, as returned by the multitemporal extraction.
        crs_dict_string (dict): Dictionary storing location codes as key and crs information as values, in dictionary form.
        geometry_column (str): field storing the geometry column. If in string form (as loaded from a csv), it will be converted to Point objects. Default='coordinates'.
        mode (str): If 'distance'(Default), compute spatial weight matrix using a distance-band kernel, specified in distance_value parameter.
                                        If 'knn', spatial weight matrix uses a specified (k_value parameter) of k number closest points to compute weights.
                                        if 'idw', Inverse Distance Weigthing is used with the specified decay power (decay parameter) to compute weight.

        distance_value (int): values in meters (crs must be projected) used as distance band for neigthours definition in distance weight matrix computation.
        decay (int): power of decay to use with IDW.
        k_value (int): number of closest points for neigthours definition in distance weight matrix computation.


    Returns:
        lisa_df (pd.DataFrame): Dataframe with the fdr threshold, local moran-s Is, p and z values and the quadrant in which each observation falls in a Moran's scatter plot.
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

    # check whether a geometry type column is present
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

                dist_mode = "k"
                decay = 0
                dist = k_value

                dist_w = weights.distance.KNN.from_dataframe(df=gdf_input, k=int(k_value))

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
