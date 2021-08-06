"""Outils module."""

import os
import re
import random
import glob
from datetime import datetime

import numpy as np
import pandas as pd
import geopandas as gpd
import seaborn as sb
from tqdm.notebook import tqdm_notebook as tqdm
from fuzzywuzzy import fuzz
import itertools
import math
from shapely.ops import split, snap
from shapely.geometry import Point, Polygon,LineString

import rasterio as ras
from rasterio import features
# from astropy.stats import median_absolute_deviation


def test_format(filename, loc_search_dict):
    """
    It returns True if the filename matches the required format (regx) or False if it doesn't.

    Args:
        filenames (str): filename to test, of the type "Seaspray_22_Oct_2020_GeoTIFF_DSM_GDA94_MGA_zone_55.tiff".
        loc_search_dict (dict): a dictionary where keys are the location codes and values are lists containing the expected full location string (["Warrnambool", "warrnambool","warrny"]).
    Returns:
        bool
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
    """
    It finds the date and returns True or a formatted version of it, from a filename of the type "Seaspray_22_Oct_2020_GeoTIFF_DSM_GDA94_MGA_zone_55.tiff".

    Args:
        filenames (str): filename to test, of the type "Seaspray_22_Oct_2020_GeoTIFF_DSM_GDA94_MGA_zone_55.tiff".
        list_months (list): expected denominations for the months. Default to ['jan','feb','mar',...,'dec'].
        to_rawdate (bool): True to format the found date into raw_date (20201022). False, return True if the date is found or False if not.
    Returns:
        bool
    """

    re_list_months = "|".join(list_months)
    regx = rf"_\d{{2}}_({re_list_months})_\d{{4}}_"

    try:
        group = re.search(regx, filename, re.IGNORECASE).group()
        if to_rawdate == False:
            return True
        else:
            group_split = group.split("_")
            dt = datetime.strptime(
                f"{group_split[1]}{group_split[2]}{group_split[3]}", "%d%b%Y"
            )
            return dt.strftime("%Y%m%d")
    except:
        return False


def filter_filename_list(filenames_list, fmt=[".tif", ".tiff"]):
    """
    It returns a list of only specific file formats from a list of filenames.

    Args:
        filenames_list (list): list of filenames.
        fmt (list): list of formats to be filtered (DEFAULT = [".tif",".tiff"])
    Returns:
        A filtered list of filenames.
    """
    return [name for name in filenames_list if os.path.splitext(name)[1] in fmt]


def round_special(num, thr):
    """It rounds the number (a) to its closest fraction of threshold (thr). Useful to space ticks in plots."""
    return round(float(num) / thr) * thr


def coords_to_points(string_of_coords):
    """
    Function to create Shapely Point geometries from strings representing Shapely Point geometries.
    Used when loading CSV with point geometries in string type.

    Args:
        string_of_coords (str): the string version of Shapely Point geometry

    Returns:
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
    """
    Function to create unique IDs from random permutations of integers and letters from the distance, tr_id, location,
    coordinates and survey_date fields of the rgb and z tables.

    Args:
        Series (Pandas series): series having the selected fields.
        tr_id_field (str)= Field name holding the transect ID (Default="tr_id").
        loc_field (str)= Field name holding the location of the survey (Default="location").
        dist_field (str)= Field name holding the distance from start of the transect (Default="distance").
        random_state (int): Random seed.

    Returns:
        A series od unique IDs.
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
    """
    Function to create IDs indipended on the survey_date, but related to to distance, tr_id and location only.
    Equivalent to use coordinates field.

    Args:
        Series (Pandas Series): series of merged table.
        random_state (int): Random seed.
    Returns:
        A series od unique spatial IDs.
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



def polygonize_valid(
    raster_path_in, output_masks_dir, name, valid_value=255.0, out_format="GPKG"
):
    """It returns the valid data polygon masks of a raster.

    Args:
        raster_path_in (str): Path to the raster, which can be a shapefiles or geopackages.
        output_location (str): Path to the output folder.
        name (str): Name of the output polygon.
        valid_value (float): Value of valid data of the input raster mask. Default is 255.0.
        out_format ('str'): If 'GPKG' (default), the polygon is a geopackage.
        Alternatively, 'ESRI Shapefile' returns .shp files.

    Returns:
        Polygons at the specified location in the specified format.
        Geopackages are reccommended (default).

    """

    # Check if output format is shapefile or Geopackage

    if out_format == "GPKG":
        file_ext = ".gpkg"
    elif out_format == "ESRI Shapefile":
        file_ext = ".shp"

    # ___________ Open the image and extract coordinates of valid data_____________#

    with ras.open(raster_path_in) as img:

        epsg = img.crs.to_dict()
        trans = img.transform

        print(f"Computing valid data mask from dataset {img.name}.")
        msk = img.read_masks(1)

        savetxt = output_masks_dir + "\\" + name + file_ext

        print("Polygonizing valid data.")
        for shape in features.shapes(msk, transform=trans):
            value = shape[1]
            if value == valid_value:
                polygon_geom = Polygon(shape[0]["coordinates"][0])
                polygon = gpd.GeoDataFrame(index=[0], crs=epsg, geometry=[polygon_geom])
                polygon.to_file(filename=savetxt, driver=out_format)

                print(f"Done with value {value}")
            else:
                print("...")

    print(f"File {name} saved at location {savetxt}")

    return polygon


def matchup_folders(
    dir1, dir2, loc_search_dict, fmts=([".tif", ".tif"], [".tif", ".tif"])
):
    """Matches files from two folders (e.g. DSMs, orthos or GCPs) and store filenames in DataFrame.

    Args:
        dir1 (str): local path of a folder where the files are stored.
        dir2 (str): local path of the second folder to match with dir1.
        loc_search_dict (dict): a dictionary where keys are the location codes and values are lists containing the expected full location string (["Warrnambool", "warrnambool","warrny"]).
        fmts (tuple): a tuple containing dir1 list of format files to retain (e.g. [".tif",."tiff"])
        and dir2 (e.g. [".cvs"]). Default=([".tif",".tif"],[".tif",".tif"]).
    Returns:
        Dataframe containing location, raw_date and dir1 and dir2 filenames (paths).
    """

    list_dir1 = filter_filename_list(getListOfFiles(dir1), fmt=fmts[0])
    list_dir2 = filter_filename_list(getListOfFiles(dir2), fmt=fmts[1])

    loc_date_labels_dir1 = [
        extract_loc_date(file1, loc_search_dict=loc_search_dict) for file1 in list_dir1
    ]
    loc_date_labels_dir2 = [
        extract_loc_date(file2, loc_search_dict=loc_search_dict) for file2 in list_dir2
    ]

    df_1 = pd.DataFrame(loc_date_labels_dir1, columns=["location", "raw_date"])
    df_1["filename_dsm"] = list_dir1

    df_2 = pd.DataFrame(loc_date_labels_dir2, columns=["location", "raw_date"])
    df_2["filename_gcp"] = list_dir2

    return pd.merge(df_1, df_2, on=["location", "raw_date"], how="inner")


def find_skiprows(filename, keyword="Easting"):
    """Find the number of rows to skip in a .CSV based on a keyword search.

    Args:
        filename (str): Local path of .CSV file.
        keyword (str): Keyword to stop the search and return its row number (default "Easting").

    Returns:
        The number (int) of the rows to skip when reading .CSV.
    """

    skiprows = 0
    with open(filename, "r+") as file:
        for line in file:
            if keyword not in line:
                skiprows += 1
            else:
                break

    return skiprows


def open_gcp_file(csv_file, crs):
    """Open a Propeller GCP (.CSV) files and return it as a geodataframe.

    Args:
        csv_file (str): Local path of .CSV file.
        crs (str): Coordinate Reference System in the dict format (example: {'init' :'epsg:4326'})

    Returns:
        Geodataframe of GCPs.
    """

    skiprows = find_skiprows(csv_file)
    df = pd.read_csv(csv_file, skiprows=skiprows)
    df["geometry"] = [Point(x, y) for x, y in zip(df.Easting, df.Northing)]
    gcp = gpd.GeoDataFrame(df, geometry="geometry", crs=crs)

    return gcp


def timeseries_to_gdf(path_timeseries_folder,list_loc_codes):
    """Returns a Geodataframe of geometries, location and survey_dates from a folder of timeseries files.

    Args:
        path_timeseries_folder (str): Local path of the timeseries files.

    Returns:
        Geodataframe.
    """

    gcp_gdf = gpd.GeoDataFrame()

    for i in getListOfFiles(path_timeseries_folder):
        tmp = gpd.read_file(i)

        tmp_dict = {
            "geometry": tmp.geometry,
            "survey_date": getDate(i),
            "location": getLoc(i, list_loc_codes),
        }
        gdf_tmp = gpd.GeoDataFrame(tmp_dict, crs=tmp.geometry.crs)
        gcp_gdf = pd.concat([gcp_gdf, gdf_tmp], ignore_index=True)

    return gcp_gdf


def gdf_distance_matrix(gdf1, gdf2, crs={"init": "epsg:3857"}):
    """
    Calculate the distance matrix between two GeoDataFrames
    Both GeoDataFrames must have the source crs set in order to be projected.


    Parameters
    ----------
    gdf1 : geopandas.GeoDataFrame
        GeoDataFrame #1
    gdf2 : geopandas.GeoDataFrame
        GeoDataFrame #2
    crs : str or dict
        Output projection parameters, passed to geopandas. Default is {'init':'epsg:3857'}.
    Returns
    -------
    pd.DataFrame
        Distance matrix dataframe; Distances can be looked up by .loc[]
    """

    # Transform to mutual coordinate system to calculate distance
    dset1 = gdf1.to_crs(crs)
    dset2 = gdf2.to_crs(crs)

    # List of coordinate pairs [x,y] for each dataset
    dset1_xy = dset1.apply(lambda b: [b.geometry.x, b.geometry.y], axis=1).tolist()
    dset2_xy = dset2.apply(lambda b: [b.geometry.x, b.geometry.y], axis=1).tolist()

    return pd.DataFrame(
        distance_matrix(dset1_xy, dset2_xy), columns=dset2.index, index=dset1.index
    )


def getCrs_from_raster_path(ras_path):
    """Returns the EPSG code of the input raster (geotiff).

    Args:
        ras_path (str): Path of the raster.

    Returns:
        EPSG code of the input raster.
    """
    with ras.open(r"{}".format(ras_path)) as raster:
        return raster.crs.to_epsg()

def filepath_raster_type(ras_path):
        """Returns 'dsm' if ras_path is of a DSM (1 single band) or 'ortho' if has multiple bands."""
        with ras.open(r"{}".format(ras_path)) as raster:
            if raster.count==1:
                return "dsm"
            else:
                return "ortho"

def getCrs_from_transect(trs_path):
    """Returns the EPSG code of the input transect file (geopackage).

    Args:
        trs_path (str): Path of the transect file.

    Returns:
        EPSG code of the input transect file.
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
    """
    Returns a dataframe with location, raw_date, filenames (paths)
    and CRS of each raster and associated transect files. Used to double-check.

    Args:
        dirNameDSM (str): Path of the directory containing the geotiffs datasets (.tiff or .tif).
        dirNameTrans (str): Path of the directory containing the transects (geopackages, .gpkg).
        loc_search_dict (list): Dictionary used to match filename with right location code.
        list_loc_codes (list): list of strings containing location codes.
        print_info (bool): If True, prints count of datasets/location and total. Default = False.

    Returns:
        Dataframe and information about raster-transect files matches.
    """

    ras_type_dict={0:"dsm",
                  1:"ortho"}

    # DF transects
    list_transects = glob.glob(f"{dirNameTrans}\*.gpkg")
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

        list_rasters = glob.glob(f"{path_in}\*.ti*")
        raster_types=[filepath_raster_type(i) for i in list_rasters]

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

        print(f"\nNUMBER OF DATASETS TO PROCESS: {counts.raw_date.sum()}")

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
    deltas=[(datetime.strptime(str(d_to), fmt) - datetime.strptime(str(d_from), fmt)).days
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
    """Helper function to split transects geometry normal to shoreline, retaining only their left (default) or right side."""

    side_dict = {"left": 0, "right": 1}
    snapped = snap(geom, geom.centroid, 0.001)
    result = split(snapped, geom.centroid)
    return result[side_dict[side]]


def create_transects(baseline, sampling_step, tick_length, location, crs, side="both"):
    """Creates a GeoDataFrame with transects normal to the baseline, with defined spacing and length.

    Args:
        baseline (gdf): baseline geodataframe.
        sampling_step (int,float): alognshore spacing of transects in the CRS reference unit.
        tick_length (int,float): transects length
        location (str): location code
        crs (): coordinate reference system to georeference the transects. It must be in dictionary form.
        side ("both", "left", "right"): If "both", the transects will be centered on the baseline. If "left" or "right", transects will start from the baseline and extend to the left/right of it.
    Returns:
        Geodataframe of transects.
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
