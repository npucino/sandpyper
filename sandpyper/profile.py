from rasterio.windows import Window, from_bounds
from rasterio.transform import rowcol
import rasterio as ras
import numpy as np
import richdem as rd
from shapely.geometry import Point
import pandas as pd
import geopandas as gpd
from tqdm import tqdm_notebook as tqdm

import os
import time
import warnings

from sandpyper.outils import create_id, filter_filename_list, getListOfFiles, getDate, getLoc


def get_terrain_info(x_coord, y_coord, rdarray):
    """
    Returns the value of the rdarray rasters.

    Args:
        x_coord, y_coord (float): Projected coordinates of pixel to extract value.
        rdarray (rdarray): rdarray dataset.

    Returns:
        rdarray pixel value.
    """


    geotransform = rdarray.geotransform

    xOrigin = geotransform[0]                         # top-left X
    yOrigin = geotransform[3]                         # top-left y
    pixelWidth = geotransform[1]                      # horizontal pixel resolution
    pixelHeight = geotransform[5]                     # vertical pixel resolution
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
        x_coord, y_coord (float): Projected coordinates of pixel to extract value.
        raster (rasterio open file): Open raster object, from rasterio.open(raster_filepath).
        bands (int): number of bands.
        transform (Shapely Affine obj): Geotransform of the raster.
    Returns:
        raster pixel value.
    """
    elevation = []
    row, col = rowcol(transform, x_coord, y_coord, round)

    for j in np.arange(bands):                  # we could iterate thru multiple bands

        try:
            data_z = raster.read(1, window=Window(col, row, 1, 1))
            elevation.append(data_z[0][0])
        except BaseException:
            elevation.append(np.nan)

    return elevation


def get_raster_px (x_coord, y_coord, raster, bands=None, transform=None):

    if isinstance(raster, richdem.rdarray):
        transform = rdarray.geotransform

        xOrigin = transform[0]                         # top-left X
        yOrigin = transform[3]                         # top-left y
        pixelWidth = transform[1]                      # horizontal pixel resolution
        pixelHeight = transform[5]                     # vertical pixel resolution
        px = int((x_coord - xOrigin) / pixelWidth)  # transform geographic to image coords
        py = int((y_coord - yOrigin) / pixelHeight)  # transform geographic to image coords

        try:
            return rdarray[py, px]
        except BaseException:
            return np.nan

    else:
        if bands == None:
            bands=raster.count()

        if bands==1:
            try:
                px_data = raster.read(1, window=Window(col, row, 1, 1))
                return px_data[0][0]
            except BaseException:
                return np.nan
        elif bands > 1:
            px_data=[]
            for band in range(1,bands+1):
                try:
                    px_data_band = raster.read(band, window=Window(col, row, 1, 1))
                    px_data.append(px_data_band[0][0])
                except BaseException:
                    px_data.append(np.nan)

            return px_data


def get_profiles(
        dsm,
        transect_file,
        transect_index,
        step,
        location,
        date_string,
        add_xy=False,
        add_terrain=False):
    """
    Returns a tidy GeoDataFrame of profile data, extracting raster information
    at a user-defined (step) meters gap along each transect.

    Args:
        dsm ():
        transect_file:
        transect_index:
        step
        location
        date_string
        add_xy (bool): True to add X and Y fields.
        add_terrain (bool): True to add slope in degrees. Default to False.

    Returns:
        gdf (GeoDataFrame) : Profile data extracted from the raster.
    """

    ds = ras.open(dsm, 'r')
    bands = ds.count                      # get raster bands. One, in a classic DEM
    transform = ds.transform            # get geotransform info

    # index each transect and store it a "line" object
    line = transect_file.loc[transect_index]
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
    tr_count = 0       # a variable used as "counter", to stop the FOR loop
    # when has gone thru all the transects

    for currentdistance in np.arange(0, int(length_m), step):

        # creation of the point on the line
        point = line.geometry.interpolate(currentdistance)
        xp, yp = point.x, point.y    # storing point xy coordinates into xp,xy objects, respectively
        x.append(xp)  # see below
        y.append(yp)  # append point coordinates to previously created and empty x,y lists
        # extraction of the elevation value from DSM
        z.append(get_elevation(xp, yp, ds, bands, transform)[0])
        if str(type(add_terrain)) == "<class 'richdem.rdarray'>":
            slopes.append(get_terrain_info(xp, yp, add_terrain))
        else:
            pass

        # append the distance value (currentdistance) to distance list
        distance.append(currentdistance)
        tr_count += 1  # increment by 1 the counter, and repeat!

    # Below, the empty lists tr_id_list and the date_list will be filled by strings
    # containing the transect_id of every point as stored in the original dataset
    # and a label with the date as set in the data setting section, after the input.

    tr_id_list = []
    date_list = []
    tr_counter = 0  # same mechanism as previous FOR loop

    while tr_counter <= tr_count:
        tr_id_list.append((int(line.name)))
        date_list.append(str(date_string))
        tr_counter += 1

    # Here below is to combine distance, elevation, profile_id and date into
    # an array first (profile_x_z), then multiple Pandas series.

    if str(type(add_terrain)) == "<class 'richdem.rdarray'>":
        profile_x_z = tuple(zip(distance, z, tr_id_list, date_list, slopes))

        ds1 = pd.Series((v[0] for v in profile_x_z))
        ds2 = pd.Series((v[1] for v in profile_x_z))
        ds3 = pd.Series((v[2] for v in profile_x_z))
        ds4 = pd.Series((v[3] for v in profile_x_z))
        ds5 = pd.Series((v[4] for v in profile_x_z))

        df = pd.DataFrame({"distance": ds1, "z": ds2, "tr_id": ds3,
                           "raw_date": ds4, "slope": ds5})

    else:
        profile_x_z = tuple(zip(distance, z, tr_id_list, date_list))

        ds1 = pd.Series((v[0] for v in profile_x_z))
        ds2 = pd.Series((v[1] for v in profile_x_z))
        ds3 = pd.Series((v[2] for v in profile_x_z))
        ds4 = pd.Series((v[3] for v in profile_x_z))

        df = pd.DataFrame({"distance": ds1, "z": ds2, "tr_id": ds3, "raw_date": ds4})

    # Here finally a Pandas dataframe is created containing all the Series previously created
    # and coordinates of the points are added to a new column called "coordinates".
    # At last, we create a Pandas GeoDataFrame and set the geometry column = coordinates

    df['coordinates'] = list(zip(x, y))
    df['coordinates'] = df['coordinates'].apply(Point)
    df['location'] = location
    df['survey_date'] = pd.to_datetime(
        date_string,
        yearfirst=True,
        dayfirst=False,
        format='%Y%m%d')
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


def get_dn(x_coord, y_coord, raster, bands, gt):
    """
    Returns the value of the raster at a specified location and band.

    Args:
        x_coord, y_coord (float): Projected coordinates of pixel to extract value.
        raster (rasterio open file): Open raster object, from rasterio.open(raster_filepath).
        bands (int): number of bands.
        transform (Shapely Affine obj): Geotransform of the raster.
    Returns:
        raster pixel value.
    """
    # Let's create an empty list where we will store the elevation (z) from points
    # With GDAL, we extract 4 components of the geotransform (gt) of our north-up image.

    dn_val = []
    xOrigin = gt[0]                         # top-left X
    yOrigin = gt[3]                         # top-left y
    pixelWidth = gt[1]                      # horizontal pixel resolution
    pixelHeight = gt[5]                     # vertical pixel resolution
    px = int((x_coord - xOrigin) / pixelWidth)  # transform geographic to image coords
    py = int((y_coord - yOrigin) / pixelHeight)  # transform geographic to image coords
    for j in range(1, 4):                  # we could iterate thru multiple bands

        band = raster.GetRasterBand(j)
        try:
            data = band.ReadAsArray(px, py, 1, 1)
            dn_val.append(data[0][0])
        except BaseException:
            dn_val.append(np.nan)
    return dn_val


def get_profile_dn(
        ortho,
        transect_file,
        transect_index,
        step,
        location,
        date_string,
        add_xy=False):

    ds = ras.open(ortho, GA_ReadOnly)

    bands = ds.count

    transform = ds.transform

    line = transect_file.loc[transect_index]

    length_m = line.geometry.length

    x = []
    y = []
    dn = []
    distance = []
    for currentdistance in np.arange(0, int(length_m), step):
        # creation of the point on the line
        point = line.geometry.interpolate(currentdistance)
        xp, yp = point.x, point.y    # storing point xy coordinates into xp,xy objects, respectively
        x.append(xp)  # see below
        y.append(yp)  # append point coordinates to previously created and empty x,y lists
        dn.append(get_dn(xp, yp, ds, bands, transform))

        distance.append(currentdistance)

    dn1 = pd.Series((v[0] for v in dn))
    dn2 = pd.Series((v[1] for v in dn))
    dn3 = pd.Series((v[2] for v in dn))
    df = pd.DataFrame({"distance": distance, "band1": dn1, "band2": dn2, "band3": dn3})
    df['coordinates'] = list(zip(x, y))
    df['coordinates'] = df['coordinates'].apply(Point)
    df['location'] = location
    df['survey_date'] = pd.to_datetime(date_string, format='%Y%m%d')
    df['tr_id'] = transect_index
    gdf_rgb = gpd.GeoDataFrame(df, geometry="coordinates")

    # Last touch, the proj4 info (coordinate reference system) is gathered with
    # Geopandas and applied to the newly created one.
    gdf_rgb.crs = str(transect_file.crs)

    #   Let's create unique IDs from the coordinates values, so that the Ids follows the coordinates
    # Transforming non-hashable Shapely coordinates to hashable strings and
    # store them into a variable

    # Transforming non-hashable Shapely coordinates to hashable strings and
    # store them into a variable

    #geometries = gdf_rgb['coordinates'].apply(lambda x: x.wkt).values

    # Let's create unique IDs from the coordinates values, so that the Ids
    # follows the coordinates
    gdf_rgb["point_id"] = [create_id(gdf_rgb.iloc[i])
                           for i in range(0, gdf_rgb.shape[0])]

    if bool(add_xy):
        # Adding long/lat fields
        gdf_rgb["x"] = gdf_rgb.coordinates.x
        gdf_rgb["y"] = gdf_rgb.coordinates.y
    else:
        pass

    return gdf_rgb


def get_merged_table(rgb_table_path, z_table_path, add_xy=False):
    """
    Function to merge rgb and z tables, creating IDs and adding slope and curvature information.
    Optionally, is adds long and lat columns in the input CRS, if not specified during the
    extraction process.

    Warning:
        Not optimised for large dataset.
        It might take up to 1h for 2 MIO observations tables.

    Args:
        rgb_table_path (str): Full path to the rgb_table, in CSV format.
        z_table_path (str): Full path to the z_table, in CSV format.
        add_xy (bool): If True, add long and lat columns in the inut CRS. Default is False.

    Returns:
        A merged and ready to process dataframe containing z and rgb informations.
    """
    print("loading tables.")
    # Loading the tables
    rgb_table = gpd.read_file(rgb_table_path)
    print("rgb_table loaded.")

    z_table = gpd.read_file(z_table_path)
    print("z_table loaded.")

    # Rounding the distance values to 0.1 precision
    print("rounding distances.")
    rgb_table["distance"] = np.round(
        rgb_table.loc[:, "distance"].values.astype("float"), 2)
    z_table["distance"] = np.round(z_table.loc[:, "distance"].values.astype("float"), 2)

    # Re-generate Shapely geometries from CSV
    print("re-generating Point geometries.")
    rgb_table['geometry'] = rgb_table.coordinates.apply(coords_to_points)
    z_table['geometry'] = z_table.coordinates.apply(coords_to_points)

    print("creating IDs.")
    # Creating unique IDs based on spatiotemporal attributes, which are the
    # same for rgb and z tables
    rgb_table["point_id"] = [
        create_id(
            rgb_table.iloc[i]) for i in range(
            0, rgb_table.shape[0])]
    z_table["point_id"] = [create_id(z_table.iloc[i])
                           for i in range(0, z_table.shape[0])]

    print("merging tables based on unique IDs. Might take a while ...")
    # Joining the rgb and z tables based on the newly created IDs to obtain
    # one single big table
    data_merged = pd.merge(z_table, rgb_table, on="point_id", validate="one_to_one")
    print("successfull merging.")

    print("cleaning dataframe and adding slope and curvature.")
    # Cleaning the data and addign slope and curvature information
    data_merged = data_merged.replace("", np.NaN)
    data_merged['z'] = data_merged.z.astype("float")
    data_merged["slope"] = np.gradient(data_merged.z)
    data_merged["curve"] = np.gradient(data_merged.slope)

    if bool(add_xy):
        print("adding long/lat info.")
        # Adding long/lat fields
        data_merged["x"] = data_merged.geometry.x
        data_merged["y"] = data_merged.geometry.y
    else:
        pass

    # Dropping extra columns inherited from the merge
    data_merged.drop(
        columns=[
            'field_1_x',
            'distance_y',
            'field_1_y',
            'coordinates_y',
            'location_y',
            'survey_date_y',
            'tr_id_y',
            'geometry_y'],
        inplace=True)

    # Renaming the useful columns kept after the merge
    data_merged.rename(
        columns={
            "tr_id_x": "tr_id",
            "distance_x": "distance",
            "coordinates_x": "coordinates",
            "location_x": "location",
            "survey_date_x": "survey_date",
            "geometry_x": "geometry"},
        inplace=True)

    return data_merged


def extract_from_folder(dataset_folder, transect_folder, list_loc_codes,
                        mode, sampling_step,
                        add_xy=False, add_slope=False, nan_values=-10000):
    """
    Wrapper to extract profiles from folder.

    Warning: The folders must contain the geotiffs and geopackages only.

    Args:
        dataset_folder (str): Path of the directory containing the datasets (geotiffs, .tiff).
        transect_folder (str): Path of the directory containing the transects (geopackages, .gpkg).
        mode (str): If 'dsm', extract from DSMs. If 'ortho', extracts from orthophotos.
        sampling_step (float): Distance along-transect to sample points at. In meters.
        add_xy (bool): If True, adds extra columns with long and lat coordinates in the input CRS.
        add_slope (bool): If True, computes slope raster in degrees (increased procesing time)
        and extract slope values across transects.
        nan_values (int): Value used for NoData in the raster format.
        In Pix4D, this is -10000 (Default).

    Returns:
        A geodataframe with survey and topographical or color information extracted.
    """

    # Get a list of all the filenames and path
    list_files = filter_filename_list(
        getListOfFiles(dataset_folder), fmt=[
            '.tif', '.tiff'])

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
            f'WARNING: add_terrain increases running time to up to {len(list_files)*3} minutes.')

    for dsm in tqdm(list_files):

        date_string = getDate(dsm)
        location = getLoc(dsm, list_loc_codes)

        if bool(add_slope):

            terr = rd.LoadGDAL(dsm)
            print(
                f"Computing slope DSM in degrees in {location} at date: {date_string} . . .")
            slope = rd.TerrainAttribute(terr, attrib='slope_degrees')
        else:
            slope = False

        transect_file_input = [a for a in list_trans if location in a]
        transect_file = gpd.read_file(transect_file_input[0])

        tr_list = np.arange(0, transect_file.shape[0])
        for i in tqdm(tr_list):
            if mode == 'dsm':
                temp = get_profiles(
                    dsm,
                    transect_file,
                    i,
                    sampling_step,
                    location,
                    date_string=date_string,
                    add_xy=add_xy,
                    add_terrain=slope)
            elif mode == 'ortho':
                temp = get_profile_dn(
                    dsm,
                    transect_file,
                    i,
                    sampling_step,
                    location,
                    date_string=date_string,
                    add_xy=add_xy)

            gdf = pd.concat([temp, gdf], ignore_index=True)

        counter += 1

    if counter == len(list_files):
        print('Extraction succesfull')
    else:
        print(f"There is something wrong with this dataset: {list_files[counter]}")

    end = time. time()
    timepassed = end - start

    print(
        f"Number of points extracted:{gdf.shape[0]}\nTime for processing={timepassed} seconds\nFirst 10 rows are printed below")

    if mode == 'dsm':
        nan_out = np.count_nonzero(np.isnan(np.array(gdf.z).astype('f')))
        nan_raster = np.count_nonzero(gdf.z == nan_values)
        gdf.z.replace(-10000, np.nan, inplace=True)

    elif mode == "ortho":
        nan_out = np.count_nonzero(
            np.isnan(np.array(gdf[["band1", "band2", "band3"]]).astype('f')))
        nan_raster = np.count_nonzero(gdf.band1 == nan_values)
        gdf.band1.replace(0.0, np.nan, inplace=True)
        gdf.band2.replace(0.0, np.nan, inplace=True)
        gdf.band3.replace(0.0, np.nan, inplace=True)

    print(
        f"Number of points outside the raster extents: {nan_out}\nThe extraction assigns NaN.")
    print(
        f"Number of points in NoData areas within the raster extents: {nan_raster}\nThe extraction assigns NaN.")

    return gdf
