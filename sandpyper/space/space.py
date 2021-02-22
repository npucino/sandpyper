from sandpiper.outils import cross_ref,getListOfFiles, getDate, getLoc,getCrs_from_raster_path
from sandpiper.profile import extract_from_folder,get_profiles

import rasterio as ras
import rasterio.mask as rasmask

import os
import glob

import pandas as pd
import re
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from shapely.geometry import MultiLineString, LineString, Point, Polygon, MultiPolygon, mapping
from shapely.ops import split, snap, unary_union


import math
from math import tan, radians, sqrt
from scipy.ndimage import gaussian_filter
import scipy.signal as sig
from skimage.filters import threshold_multiotsu
from sklearn.metrics import mean_squared_error


from tqdm import tqdm

import richdem as rd
import itertools as it
from itertools import groupby
from operator import itemgetter
import datetime as dt



def grid_from_pts(pts_gdf, width, height, crs):
    """
    Create a georeferenced grid of polygones from points along a line (shoreline).
    Used to extract tiles (images patches) from rasters.

    Args:
        pts_gdf (GeoDataFrame): The geodataframe storing points along a shoreline.

        width, height (int,float): The width and heigth of each single tile of the grid, given in the CRS unit (use projected CRS).

        crs (str): Coordinate Reference System in the dictionary format (example: {'init' :'epsg:4326'})

    Returns:
        Grid : A GeoDataFrame storing polygon grids, with IDs and geometry columns.
    """


    xmin,ymin,xmax,ymax =  pts_gdf.total_bounds

    rows = int(np.ceil((ymax-ymin) /  height))
    cols = int(np.ceil((xmax-xmin) / width))

    XleftOrigin = xmin
    XrightOrigin = xmin + width
    YtopOrigin = ymax
    YbottomOrigin = ymax- height
    polygons = []
    for i in range(cols):
        Ytop = YtopOrigin
        Ybottom =YbottomOrigin
        for j in range(rows):
            polygons.append(Polygon([(XleftOrigin, Ytop), (XrightOrigin, Ytop), (XrightOrigin, Ybottom), (XleftOrigin, Ybottom)]))
            Ytop = Ytop - height
            Ybottom = Ybottom - height
        XleftOrigin = XleftOrigin + width
        XrightOrigin = XrightOrigin + width

    grid = gpd.GeoDataFrame({'grid_id':range(len(polygons)),
                             'geometry':polygons}, crs=crs)

    return grid


def add_grid_loc_coords (grid_gdf,location=None):
    """
    Add coordinate fields of the corners of each grid tiles.

    Args:
        grid_gdf (GeoDataFrame): The geodataframe storing the grid, returned by the grid_from_pts function.

        location (str): The location code associated with the grid. Defaults to None.

    Returns:
        The original grid, with UpperLeft X and Y (ulx,uly), UpperRight X and Y (urx,ury), LowerLeft X and Y (llx,llr) and LowerRigth X and Y (lrx,lry) coordinates fields added.
    """

    if location is None:
        location=np.nan

    ulxs=[]
    urxs=[]
    lrxs=[]
    llxs=[]

    ulys=[]
    urys=[]
    lrys=[]
    llys=[]

    for grid in (range(grid_gdf.shape[0])):

        coords=grid_gdf.iloc[grid].geometry.exterior.coords.xy

        # get upper-left, upper-right, lower-right and lower-left X coordinates.
        ulx=coords[0][0]
        urx=coords[0][1]
        lrx=coords[0][2]
        llx=coords[0][3]

        # get upper-left, upper-right, lower-right and lower-left Y coordinates.
        uly=coords[1][0]
        ury=coords[1][1]
        lry=coords[1][2]
        lly=coords[1][3]

        ulxs.append(ulx)
        urxs.append(urx)
        lrxs.append(lrx)
        llxs.append(llx)

        ulys.append(uly)
        urys.append(ury)
        lrys.append(lry)
        llys.append(lly)

    grid_gdf.loc[:,'ulx']=ulxs
    grid_gdf.loc[:,'urx']=urxs
    grid_gdf.loc[:,'lrx']=lrxs
    grid_gdf.loc[:,'llx']=llxs

    grid_gdf.loc[:,'uly']=ulys
    grid_gdf.loc[:,'ury']=urys
    grid_gdf.loc[:,'lry']=lrys
    grid_gdf.loc[:,'lly']=llys

    grid_gdf.loc[:,"location"]=location

    return grid_gdf


def grid_from_shore (shore,width,height,
                     location_code,crs='shore',
                     shore_res=10,
                    plot_it=True):
    """
    Create a georeference grid of equal polygones (tiles) along a line (shoreline) and select those tiles that contain at least partially the line.

    TO DO: CRS should also be a string for specific CRS. Probablt only need the first and last points endpoints of the shoreline, or can get box directly.

    Args:
        shore (geodataframe): The geodataframe storing the input line from where the grid will be created.
        width, height (int,float): The width and heigth of each single tile of the grid, given in the CRS unit (use projected CRS).
        location_code (str): The location code associated with the grid.
        crs (dict or 'shore'): If 'shore', use the same CRS of the input line. If dict, keys should be the location code and values the values crs in the dictionary format ('wbl' : {'init' :'epsg:32754'}).
        shore_res (int,float): the alongshore spacing of points plotted along the line in the CRS unit (default=10). It doesn't need to be a small value, it is used to get the extent of the bounding box that encapsulate all the shoreline, before split this into a grid.
        plot_it (bool): plot the shoreline, full grid and the tiles selected containing the lien (in red). Default to True.

    Returns:
        GeoDataFrame of the grid of only tiles containing the line.
    """

    xs=[]
    ys=[]
    points=[]

    for distance in np.arange(0,shore.length.values[0],shore_res): # shore_res: meters alongshroe to get points from shore
        pt=shore.interpolate(distance)
        points.append(pt.values[0])
        xs.append(pt.values[0].x)
        ys.append(pt.values[0].y)

    pts=[[x,y] for x,y in zip(xs,ys)]
    pts=np.array(pts)

    if isinstance(crs,dict):
        crs_in=crs[location_code]
    elif crs == 'shore':
        crs_in=shore.crs


    points_gdf=gpd.GeoDataFrame({"local_id":range(len(points)),
                        "geometry":points}, geometry="geometry", crs=crs_in)
    grid=grid_from_pts(points_gdf,width,height,crs=crs_in)


    shore_grids=grid[grid.geometry.apply(lambda x: points_gdf.geometry.within(x).any())]

    add_grid_loc_coords(shore_grids, location=location_code)


    if bool(plot_it)==True:
        f,ax=plt.subplots(figsize=(10,10))


        grid.plot(color='white', edgecolor='black', ax=ax)
        shore.plot(ax=ax)
        shore_grids.geometry.boundary.plot(color=None,edgecolor='r',linewidth = 4,ax=ax)

    else:
        pass

    return shore_grids


def dissolve_shores(gdf_shores, field='date'):
    """
    Dissolves multi-part shorelines into one geometry per location-date.
    Uses GeoPandas.GeoDataFrame.dissolve method.

    To Do: multi-location can be infered by len

    Args:
        gdf_shores (GeoDataFrame): The geodataframe storing the shoreline.
        field (str): The field to be used to dissolve shorelines. Default to "date".

    Returns:
        GeoDataFrame with one geometry per location-date combination.
    """
    if len(gdf_shores.location.unique()) != 1:
        multi_location=True
    else:
        multi_location=False

    dissolved=pd.DataFrame()
    if bool(multi_location)==True:
        for loc in gdf_shores.location.unique():
            geom=gdf_shores.query(f"location=='{loc}'").dissolve(by=field).reset_index()
            dissolved=pd.concat([dissolved,geom], ignore_index=True)
    else:
        gdf_shores["diss"]=0
        geom=gdf_shores.dissolve(by="diss").reset_index()
        dissolved=pd.concat([dissolved,geom], ignore_index=True)
        dissolved.drop("diss", axis=1,inplace=True)

    return dissolved


def tiles_from_grid (grid,img_path,output_path, list_loc_codes, mode='rgb', geotransform=True,sel_bands=None,driver="PNG"):
    """
    Returns a dataframe with location, raw_date, filenames (paths) or geopackage index and CRS of each raster and its associated vector files.
    If the directory containing the vector files has ony one file, it is assumed that this file stores vectors
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

    if geotransform== 'only':
        geo_dict=dict()
        export_image=False
        export_geo_dict=True

    elif bool(geotransform)==True:
        geo_dict=dict()
        export_image=True
        export_geo_dict=True
    elif bool(geotransform)==False:
        export_image=True
        export_geo_dict=False



    with ras.open(img_path,'r') as dataset:


        if mode=='rgb':

            count=3
            source_idx=[1,2,3] # the band indices of the source image
            out_idx=source_idx
            sel_bands=None

            height_idx=1
            width_idx=2

        elif mode=='multi':

            if driver=="PNG":
                print("NOTE: PNG format doesn't support multibands. Returning GeoTiffs instead.")
                driver="GTiff"
            else:
                pass

            sel_bands=None
            count=dataset.count
            source_idx=list(dataset.indexes)
            out_idx=None
            height_idx=1
            width_idx=2

        elif mode=='mask':

            count=1
            source_idx=1
            out_idx=1
            height_idx=0
            width_idx=1

        elif mode=='custom':
            if len(sel_bands)>3 and driver=="PNG":
                print("NOTE: More than 3 bands selected for the creation of PNG images. PNG format doesn't support multibands. Returning GeoTiffs instead.")
                driver="GTiff"
            else:
                pass

            source_idx=sel_bands
            out_idx=None
            count=len(sel_bands)
            height_idx=1
            width_idx=2

        for i in range(grid.shape[0]):
            mask=grid.iloc[[i]]
            geom=Polygon(   # create the square polygons to clip the raster with

       ((mask.iloc[0]['ulx'],mask.iloc[0]['uly']),
       (mask.iloc[0]['urx'],mask.iloc[0]['ury']),
       (mask.iloc[0]['lrx'],mask.iloc[0]['lry']),
       (mask.iloc[0]['llx'],mask.iloc[0]['lly']),
        (mask.iloc[0]['ulx'],mask.iloc[0]['uly']))

            )

            try:
                out_image, out_transform = rasmask.mask(dataset, [mapping(geom)], crop=True,filled=False,
                                                   indexes=source_idx)
                out_meta = dataset.meta


                # format name: LocationCode_GridId_Date.tif
                if driver=="PNG":
                    ext='png'
                elif driver=='GTiff':
                    ext='tif'
                else:
                    raise NameError("Driver must be either 'PNG' or 'GTiff'.")


                if driver=="PNG":
                    ext='png'
                    out_meta.update({
                    'dtype':ras.uint16})
                else:
                    ext='tif'


                out_meta.update({"driver": driver,
                 "height": out_image.shape[height_idx],
                 "width": out_image.shape[width_idx],
                 "transform": out_transform,
                "count":count})

                tile_name=f"{getLoc(img, list_loc_codes)}_{mask.iloc[0]['grid_id']}_{getDate(img)}_{mode}"
                savetxt=os.path.join(output_path,f"{tile_name}.{ext}")

                if bool(export_image)==True:
                    with ras.open(savetxt, "w", **out_meta) as dest:
                        if driver=="PNG":
                            dest.write(out_image.astype(ras.uint16), indexes=out_idx)
                        else:
                            dest.write(out_image,indexes=out_idx)
                    print(f"Succesfully saved tile: {tile_name} .")

                    if bool(export_geo_dict)==True:
                        geo_dict.update({tile_name: out_transform})
                    else:
                        pass

                else:
                    pass

                if bool(export_geo_dict)==True:
                    geo_dict.update({tile_name: out_transform})
                else:
                    pass

            except:
                print(f"Error with tile {mask.iloc[0]['grid_id']}. Might not overlap iamge. Skipping.")

    if bool(export_geo_dict)==True:
        return geo_dict
    else:
        pass


def check_overlay (line_geometry, img_path):
    """ Evaluate wether a line intesects the extent of a raster.
        Returns True if a valid intersection is found or False if not. In case of MultiLine features,
        evaluate if any of the lines intersects with the raster extent,
        which confirms that the CRS of both shapes geometries are correctly matched.

    Args:
        line_geometry (Shapely Line or MultiLinestring objects): geometry of line to evaluate its overlay on raster.
        img_path (str): Path to the geotiff to evaluate line overlay with.

    Returns:
        True, if a valid match is found. False, if the line do not intersect the raster."""

    # create a polygone with raster bounds
    with ras.open(img_path,'r') as dataset:

        ul=dataset.xy(0,0)
        ur=dataset.xy(0,dataset.shape[1])
        lr=dataset.xy(dataset.shape[0],dataset.shape[1])
        ll=dataset.xy(dataset.shape[0],0)

        ext_poly=gpd.GeoSeries(Polygon([ul,ur,lr,ll,ul]),crs=dataset.crs)


    # get geom_type
    if isinstance(line_geometry.geom_type, str):
        geom_type=line_geometry.geom_type
    elif isinstance(line_geometry.geom_type, pd.Series):
        geom_type=geom.geom_type[0]




    if geom_type == 'MultiLineString':

        geom_in=list(line_geometry)[0]

        if ext_poly[0].intersects(geom_in):
            return True
        else:
            return False

    elif geom_type == 'LineString':

        geom_in=line_geometry

        if ext_poly[0].intersects(geom_in):
            return True
        else:
            return False
    else:
        raise IOError('Shape is not a line.')







    ## http://wikicode.wikidot.com/get-angle-of-line-between-two-points
    ## angle between two points


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


def split_transects(geom, side='left'):
    """Helper function to split transects geometry normal to shoreline, retaining only their left (default) or right side."""

    side_dict={'left':1,
    'right':0}
    snapped=snap(geom, geom.centroid, 0.001)
    result = split(snapped, geom.centroid)
    return result[side_dict[side]]


def create_transects(baseline,sampling_step,tick_length,location, crs, side='both'):
    """ Creates a GeoDataFrame with transects normal to the baseline, with defined spacing and length.

    Args:
        baseline (str): Local path of the timeseries files, as returned by the multitemporal extraction.
        list_loc_codes (list): list of strings containing location codes.
    Returns:
        Geodataframe.
    """
    #     if crs != baseline.crs:
    #         print(f"WARNING: Baseline CRS ({baseline.crs['init']}) is different to desired CRS ({crs['init']}). Reprojecting to: {crs['init']}.")
    #         baseline=baseline.to_crs(crs)
    #     else:
    #         pass

    if side!='both':
        tick_length=2*tick_length
    else:
        pass

    try:
        dists=np.arange(0, baseline.geometry.length[0], sampling_step)
    except:
        try:
            dists=np.arange(0, baseline.geometry.length, sampling_step)
        except:
            dists=np.arange(0, baseline.geometry.length.values[0], sampling_step)

    points_coords=[]
    try:
        for j in [baseline.geometry.interpolate(i) for i in dists]:
            points_coords.append((j.geometry.x[0], j.geometry.y[0]))
    except:
        for j in [baseline.geometry.interpolate(i) for i in dists]:
            points_coords.append((j.geometry.x, j.geometry.y))


            # create transects as Shapely linestrings

    ticks=[]
    for num, pt in enumerate(points_coords, 1):
            ## start chainage 0
            if num == 1:
                angle = getAngle(pt, points_coords[num])
                line_end_1 = getPoint1(pt, angle, tick_length/2)
                angle = getAngle([line_end_1.x,line_end_1.y], pt)
                line_end_2 = getPoint2([line_end_1.x,line_end_1.y], angle, tick_length)
                tick = LineString([(line_end_1.x, line_end_1.y), (line_end_2.x, line_end_2.y)])

            ## everything in between
            if num < len(points_coords) - 1:
                angle = getAngle(pt, points_coords[num])
                line_end_1 = getPoint1(points_coords[num], angle, tick_length/2)
                angle = getAngle([line_end_1.x,line_end_1.y], points_coords[num])
                line_end_2 = getPoint2([line_end_1.x,line_end_1.y], angle, tick_length)
                tick = LineString([(line_end_1.x, line_end_1.y), (line_end_2.x, line_end_2.y)])

            ## end chainage
            if num == len(points_coords):
                angle = getAngle(points_coords[num - 2], pt)
                line_end_1 = getPoint1(pt, angle, tick_length/2)
                angle = getAngle([line_end_1.x,line_end_1.y], pt)
                line_end_2 = getPoint2([line_end_1.x,line_end_1.y], angle, tick_length)
                tick = LineString([(line_end_1.x, line_end_1.y), (line_end_2.x, line_end_2.y)])

            ticks.append(tick)


    gdf_transects=gpd.GeoDataFrame({'tr_id':range(len(ticks)),
                          'geometry':ticks,
                         'location':[location for i in range(len(ticks))]}, crs=crs)


                         # clip the transects

    if side =='both':
        pass
    else:

        gdf_transects["geometry"]=gdf_transects.geometry.apply(split_transects, **{'side':side})

    return gdf_transects


def correct_multi_detections (shore_pts, transects):

    geometries=[]
    for i in range(shore_pts.shape[0]):
        pt_i=shore_pts.iloc[[i]]
        if pt_i.qa.values[0] == 0:
            geometries.append(np.nan)
        else:
            if pt_i.qa.values[0] != 1:

                start_line=transects.query(f"tr_id=='{pt_i.tr_id.values[0]}'").geometry.boundary.iloc[0][0]

                min_idx=np.argmin([start_line.distance(pt) for pt in pt_i.geometry.values[0]])
                geometries.append(pt_i.geometry.values[0][min_idx])

            else:
                geometries.append(pt_i.geometry.values[0])

    return geometries


def extract_shore_pts(transects,shore,crs=32754, tr_id_field='tr_id',date_field='date',raw_date_format='%Y%m%d'):

    points=[]
    tr_ids=[]
    survey_dates=[]


    for i in range(transects.shape[0]):
        transect_i=transects.iloc[i]
        try:
            point1 = transect_i.geometry.intersection(shore.unary_union)
        except:
            try:
                point1 = transect_i.geometry.intersection(shore)
            except:
                point1 = np.nan

        points.append(point1)
        tr_ids.append(transect_i.loc[tr_id_field])
        survey_dates.append(shore[date_field].values[0])

    df=pd.DataFrame({"geometry":points,
                "tr_id":tr_ids,
                "raw_date":survey_dates})

    replace_nan=[df.iloc[i].geometry if df.iloc[i].geometry.is_empty == False else np.nan for i in range(df.shape[0])]
    qa=[0 if type(i) == float else len(i) if i.geom_type == "MultiPoint" else 1 for i in replace_nan]

    df["geometry"]=replace_nan
    df["qa"]=qa
    df["geometry"]=correct_multi_detections(df, transects)

    gdf=gpd.GeoDataFrame(df, geometry="geometry",crs=crs)
    #     gdf['date_formatted']=[dt.datetime.strptime(str(raw_date),raw_date_format) for raw_date in gdf.raw_date]

    return gdf


def shore_shift(transects,gt,sat,crs, baseline_pts,sat_from_baseline=False):

    sat_pts=extract_shore_pts(transects,sat,crs)
    gt_pts=extract_shore_pts(transects,gt, crs,date_field='raw_date')

    sat_dists=baseline_pts.distance(sat_pts)
    gt_dists=baseline_pts.distance(gt_pts)

    baseline_pts["sat_dist"]=baseline_pts.distance(sat_pts)
    baseline_pts["gt_dist"]=baseline_pts.distance(gt_pts)
    baseline_pts.dropna(subset=["sat_dist", "gt_dist"], thresh=2, inplace=True)
    rmse=sqrt(mean_squared_error(baseline_pts.gt_dist, baseline_pts.sat_dist))

    shore_shift=sat_dists - gt_dists # if negative, seaward bias.

    transects['rmse']=rmse
    transects['std']=shore_shift.std()
    transects["shore_shift"]=shore_shift

    if bool(sat_from_baseline)==True:
        transects["sat_from_baseline"]=baseline_pts.distance(sat_pts)
    else:
        pass

    return transects


def rawdate_from_timestamp_str(timestamp_str):

    splitted=timestamp_str.split(" ")[0].split("-")
    return splitted[0]+splitted[2]+splitted[1]


def corr_baseline_distance (dist, slope, z_tide):

    if isinstance(dist, (int, float, complex)):

        tan_b=tan(radians(slope))
        return dist+(z_tide/tan_b)

    elif isinstance(dist, (pd.Series)):

        tans_b=[tan(radians(i)) for i in slope]
        corrs=[ d+(z/sl) for d,sl,z in zip(dist,tans_b,z_tide)]

        return corrs
    else:
        raise TypeError("Input must be either Pandas.Series or numeric (int,float).")


def error_from_gt (shorelines, groundtruths,crs_dict_string,
                   location, sampling_step, tick_length,
                   shore_geometry_field,
                   gt_geometry_field,
                   side='both',
                   baseline_mode="dynamic", tidal_correct=None):

     # shore and gt geometry fields allow to swap between corrected or original geometry fields of both shorelines and groundtruths
     # baseline_mode: if dynamic, stats will be computed from transects created from each groundtruth shoreline.
                   # if path to a .gpkg is provided, then use those arbitrary location specific baseline and transects will
                   # be fixed.
    # crs=int(crs_dict_string[location]['init'][-5:]) # get EPSG code from crs dict
    crs=crs_dict_string[location]

    if os.path.isfile(baseline_mode):
        print("Fixed baseline mode selected.")
        baseline_loc=gpd.read_file(baseline_mode)

        if baseline_loc.crs != crs:
            baseline_loc=baseline_loc.to_crs(crs)
        else:
            pass

        transects=create_transects(baseline_loc, sampling_step=sampling_step, tick_length=tick_length,
                                       crs=crs, location=location)
        df_temp=pd.DataFrame.from_dict(
        {'geometry':[Point(tr_geom.coords[0][0],tr_geom.coords[0][1]) for tr_geom in transects.geometry],
        'tr_id': transects.tr_id})
        baseline_pts=gpd.GeoDataFrame(df_temp,geometry='geometry',crs=crs)

    elif baseline_mode =="dynamic":
        print("Dynamic baseline mode selected.")

    else:
        raise TypeError("Baseline mode must be either 'dynamic' or a valid path to a .gpkg.")


    shore_shift_df=pd.DataFrame()


    cs_shore_in=groundtruths.query(f"location=='{location}'") # subset CS shorelines with a location
    tests=shorelines.query(f"location=='{location}'") # select all Sentinel-2 shorelines in that location

    # if cs_shore_in.crs != crs:
    #     cs_shore_in=cs_shore_in.to_crs(crs)
    # else:
    #     pass
    #
    # if tests.crs != crs:
    #     tests=tests.to_crs(crs)
    # else:
    #     pass


    #     tests["raw_date"]=tests.timestamp.apply(rawdate_from_timestamp_str) # DEA

    for i in range(cs_shore_in.shape[0]): # for all CS groundtruths in location

        groundtruth=cs_shore_in.iloc[[i]] # select the current groundtruth

        survey_date=groundtruth.raw_date.values[0] # get survey date           # HARDCODED DATE FIELD! BAD

        if survey_date in tests.raw_date.unique():
            print(f"Working on {survey_date}...")

        else:
            print(f"Groundtruth in date {survey_date} not matched with any shorelines date.")




        if baseline_mode =="dynamic":
            # create transects and baselines pts dynamically from each groundtruth shoreline
            transects=create_transects(groundtruth, sampling_step=sampling_step, tick_length=tick_length,
                                       crs=crs, location=location)

            # create a geodataframe of transects starting points to compute distance from
            df=pd.DataFrame.from_dict(
            {'geometry':[Point(tr_geom.coords[0][0],tr_geom.coords[0][1]) for tr_geom in transects.geometry],
            'tr_id': transects.tr_id})
            baseline_pts=gpd.GeoDataFrame(df,geometry='geometry',crs=crs)

            # extract groundtruth distance from baseline (should be half transect, i.e. transect centroid)
            gt_pts=extract_shore_pts(transects,groundtruth)




        transects.loc[:,"raw_date"]=survey_date


        # list all the satellite shorelines that corresponds to the groundtruth
        sat_shores_in=tests.query(f"raw_date=='{survey_date}'")


        for j in range(sat_shores_in.shape[0]):

            shore_sat=sat_shores_in.iloc[[j]]

            new_transects=shore_shift(transects=transects,
                                  gt=groundtruth,
                                  sat=shore_sat,
                                  crs=crs,
                                  baseline_pts=baseline_pts,
                                  sat_from_baseline=True)

            shore_sat.rename({"geometry":"geom_shoreline"},axis=1, inplace=True)
            merged=pd.merge(shore_sat,new_transects,on="raw_date")
            merged.rename({"location_x":"location"},axis=1, inplace=True)


            shore_shift_df=pd.concat([shore_shift_df,merged], ignore_index=True)

    return shore_shift_df


def toes_from_slopes(series,distance_field="distance",slope_field="slope", sigma=0, peak_height=30):
    """Returns candidate distances to clip beachfaces based on satellite or CS shoreline and these values."""

    sorted_series=series.sort_values([distance_field])

    gauss=gaussian_filter(sorted_series.loc[:,slope_field], sigma=sigma)
    peak=sig.find_peaks(gauss,height=peak_height,)

    try:
        toe_distances=sorted_series.iloc[peak[0]][distance_field]

    except:
        toe_distances=np.nan

    return toe_distances


def toes_candidates(df,location_field='location',date_field="raw_date",
             tr_id_field="tr_id",
             distance_field="distance",
             slope_field="slope",
             sigma=0, peak_height=30):

    """Returns candidate distances to clip beachfaces based on satellite or CS shoreline and these values."""

    apply_dict={'distance_field':distance_field,
               'slope_field':slope_field,
               'sigma':sigma,
               'peak_height':peak_height}

    dist_toe_=df.groupby([location_field,date_field,tr_id_field]).apply(toes_from_slopes,**apply_dict)

    dist_toe_df=pd.DataFrame(pd.Series(dist_toe_, name="toe_distances"))

    df_formatted=pd.DataFrame(dist_toe_df.reset_index().groupby([location_field,date_field,tr_id_field])["toe_distances"].apply(np.array))

    return df_formatted


def consecutive_ids (data, indices=True, limit=1):
    """Returns indices of consecutive tr_ids in groups. Covenient to create multi-line geometries in case of disconnected shorelines.

    Args:
        data (): .
        indices (bool): .
        limit (int): Default=1.
    Returns:
        Groups.

    """

    if bool(indices)==False:
        return_i=1 # return data values
    else:
        return_i=0

    groups=[]

    for k, g in groupby(enumerate(data), lambda ix : ix[0] - ix[1]):
     groups.append(list(map(itemgetter(return_i), g)))

    groups= [i for i in groups if len(i) > limit]

    return groups


def tidal_correction(shoreline,cs_shores,gdf,baseline_folder,crs_dict_string,
                     limit_correction, mode,  alongshore_resolution,slope_value='median',
                     side='both', tick_length=200, subset_loc=None, limit_vertex=1,
                     baseline_threshold='infer',replace_slope_outliers=True,
                     save_trs_details=False, trs_details_folder=None,
                     gdf_date_field="raw_date",distance_field='distance',
                     date_field='raw_date', # of transect geodataframe
                     toe_field='toe'):

    """
    Simple tidal correction for input shorelines. It can automatically extract subaerial beachfaces and more.

    Args:
        shoreline (GeoDataFrame): The geodataframe storing points along a shoreline.

        cs_shores (GeoDataFrame): The width and heigth of each single tile of the grid, given in the CRS unit (use projected CRS).

        gdf (GeoDataFrame): Coordinate Reference System in the dictionary format (example: {'init' :'epsg:4326'})

        baseline_folder: Path to the folder storing the baseline Geopackages (.gpkgs).

        crs_dict_string: Dictionary storing location codes as key and crs information as values, in dictionary form.
        Example: crs_dict_string = {'wbl': {'init': 'epsg:32754'},
                   'apo': {'init': 'epsg:32754'},
                   'prd': {'init': 'epsg:32755'},
                   'dem': {'init': 'epsg:32755'} }

        limit_correction (bool): If True, only use beachface slopes to compute the statistic for tidal correction.
        When False, retain the full transects to compute the slope statistics to correct shorelines. When a slope value is
        provided, it automatically sets to False. Defaults to True.

        mode (str, 'sat' or 'gt'): If 'sat', use satellite shorelines as seaward edge to classify beachfaces.
        If 'gt', use groundthruth shorelines instead.

        slope_value (int,float,'mean','median','min','max'): If a numeric value is provided (assumed to be in degrees),
        use it to correct the shorelines. If one of 'mean','median','min','max', use this statistics instead.
        It also computes range, standard deviation and variance for
        analytical purposes, despite should not be used to correct shorelines.

        side (str, 'left', 'right', 'both'): Wether if retain only the left, right or both sides of the transects once created.
        Defaults to 'both'.

        alongshore_resolution ('infer', int, float): The alongshore spacing between transects, in the unit of measure of the
        location CRS. If 'infer', use the gdf file to detect the spacing with 10cm precision. If the transects
        spacing is less than 10cm, set the spacing manually.
        Note: It also smoothes the original line if this value is greater of the original line vertex spacing.

        tick_length (int, float): Across-shore length of each transect in the unit of measure of the location CRS.

        limit_vertex (int): Sets the minimum number of consecutive transect ids to create one segment
        of the corrected shoreline. Defaults to 1.

        baseline_threshold:

        replace_slope_outliers (bool): If True (default), replace the values of the defined slope statistics (slope_value parameter)
        with its survey-level median.

        save_trs_details (bool): True to save. It defaults to False.

        trs_details_folder (str): Folder where to save the transect details. Defaults to None.

        gdf_date_field (str): Date field of the slope geodataframe used to correct the shoreline.Defaults to "raw_date".

    Returns:
        GeoDataFrame containing two geometry columns :
    """


    if isinstance(slope_value, (int,float)):
        limit_correction=False
        extract_slope=False
        print(f"User-defined slope value of {slope_value} degrees will be used to correct shorelines. No beachfaces extracted.")

    else:
        limit_correction=True
        extract_slope=True


    if limit_correction:
        limited='limited'
    else:
        limited="notlimited"

    if "water_index" in shoreline.columns:
        dataset_type='uavs_hores'
    else:
        dataset_type='satellite_shores'


    grouping_fields=set(['water_index','thr_type','raw_date','location'])
    cols=set(shoreline.columns.values)

    group_by_fields=list(cols.intersection(grouping_fields))


    #__________________ TIDAL CORRECTION__________________________________________________________________

    big_final=pd.DataFrame()


    list_locations=gdf.location.unique() # list the locations in the GDF file

    for location in list_locations:   # work iteratively on each location
        print(f"working on {location}")

        if isinstance(alongshore_resolution, (int,float)):
            print(f"Transect spacing set manually (alongshore resolution) = {alongshore_resolution} .")
        elif alongshore_resolution=='infer':

            alongshore_resolution=infer_along_trs_spacing(gdf)
            print(f"Transect spacing (alongshore resolution) inferred = {alongshore_resolution} .")
        else:
            raise NameError("Alongshore resolution must be either a float, int, or 'infer'.")


        list_dates=gdf.query(f"location=='{location}'").raw_date.unique() # get dates of location
        shores_to_corr=shoreline.query(f"location=='{location}' & raw_date in @list_dates") # get shores to correct
        gt_shores=cs_shores.query(f"location=='{location}' & raw_date in @list_dates") # get groudtruths shores
        crs=crs_dict_string[location] # get crs of location

        if shores_to_corr.crs != crs:
            shores_to_corr=shores_to_corr.to_crs(crs)
        else:
            pass

        if gt_shores.crs != crs:
            gt_shores=gt_shores.to_crs(crs)
        else:
            pass


        # read baseline as a GeoDataFrame
        try:
            baseline_location_path=glob.glob(f"{baseline_folder}/{location}*.gpkg")[0]
        except:
            raise NameError("Baseline file not found.")
        baseline=gpd.read_file(baseline_location_path, crs=crs)


        # create transects  # same parameter as the slope extraction. TO DO: if exists, use existing data.
        transects=create_transects(baseline=baseline, side=side,
                                       sampling_step=alongshore_resolution, tick_length=tick_length,
                                       location=location, crs=crs)



        # 1)_____________________________________________CREATE THE BASELINE POINTS______________________

        # create a geodataframe of transects starting points to compute distance from

        df_tmp=pd.DataFrame.from_dict(
        {'geometry':[Point(tr_geom.coords[0][0],tr_geom.coords[0][1]) for tr_geom in transects.geometry],
         'tr_id': transects.tr_id,
        'location':location})
        baseline_pts=gpd.GeoDataFrame(df_tmp,geometry='geometry',crs=crs)


        # loop through all transects in each satellite shoreline to compute distances from the baseline points
        print("Computing cross-shore distances of satellite shorelines.")

        orig_shore_base_distances=pd.DataFrame()

        for i in tqdm(range(shores_to_corr.shape[0])):
            # if "water_index" column is present, we will need to group based on water indices and threshold types too


            original_shore=shores_to_corr.iloc[[i]][[*group_by_fields,"geometry"]]

            original_shore_pts=extract_shore_pts(transects,original_shore,date_field='raw_date', crs=crs)

            matched=pd.merge(original_shore_pts,baseline_pts,how='left',on='tr_id', suffixes=("_pt_on_shore","_pt_on_base"))
            dists=[]
            for shore_pt, base_pt in zip(matched.geometry_pt_on_shore,matched.geometry_pt_on_base):
                try:
                    dists.append(shore_pt.distance(base_pt))
                except:
                    dists.append(np.nan)
            matched["sat_from_baseline"]=dists
            matched["raw_date"]=original_shore.raw_date.values[0]

            if "water_index" in shoreline.columns:

                matched["water_index"]=original_shore.water_index.values[0]
                matched["thr_type"]=original_shore.thr_type.values[0]
            else:
                pass

            orig_shore_base_distances=pd.concat([matched.dropna(subset=['sat_from_baseline']),
                                                 orig_shore_base_distances], ignore_index=True)

        # update transects with the baseline shore distances info
        orig_shore_base_distances=pd.merge(orig_shore_base_distances,transects[['tr_id','geometry']],on='tr_id', how='left')
        shores_to_corr.raw_date.astype(int)



        # loop through all transects in each groundthruth shoreline to compute distances from the baseline points
        print("Computing cross-shore distances of UAV shorelines.")

        if mode=='gt':   # we will use these points to limit the beachface extraction

            gt_base_distances=pd.DataFrame()

            for shoreline_i in tqdm(range(gt_shores.shape[0])):
                shore_i=gt_shores.iloc[[shoreline_i]][["location","raw_date","geometry"]]
                shore_pts_gt=extract_shore_pts(transects,shore_i,date_field='raw_date')

                matched=pd.merge(shore_pts_gt,baseline_pts,how='left',on='tr_id', suffixes=("_gt_on_shore","_gt_on_base"))
                dists=[]
                for shore_pt, base_pt in zip(matched.geometry_gt_on_shore,matched.geometry_gt_on_base):
                    try:
                        dists.append(shore_pt.distance(base_pt))
                    except:
                        dists.append(np.nan)
                matched["gt_from_baseline"]=dists
                matched["raw_date"]=shore_i.raw_date.values[0]

                gt_base_distances=pd.concat([matched.dropna(subset=['gt_from_baseline']),gt_base_distances], ignore_index=True)

            gt_base_distances=pd.merge(gt_base_distances,transects[['tr_id','geometry']],on='tr_id', how='left')
        else:
            pass


        #_______ EXTRACT BEACHFACE BASED ON SHORELINE POSITION AND DUNE TOE______________________________

        if bool(limit_correction) == True:

            print(f"Extracting dune toes from slope profiles.")
            gdf_loc=gdf.query(f"location=='{location}'")

            toes_cand=toes_candidates(gdf_loc, date_field='raw_date')

            if mode=='gt':

                baseline_distances_in=gt_base_distances
                baseline_field="gt_from_baseline"
                txt='UAV-derived'

                base_dist_toes=pd.merge(baseline_distances_in,toes_cand.reset_index(), on=["location","raw_date","tr_id"], how="left")


            else:

                baseline_distances_in=orig_shore_base_distances
                baseline_field="sat_from_baseline"
                txt='satellite-derived'

                base_dist_toes=pd.merge(baseline_distances_in,toes_cand.reset_index(), on=["location","raw_date","tr_id"], how="left")

            toes=[]
            for t,d in zip(base_dist_toes.loc[:,"toe_distances"],base_dist_toes.loc[:,baseline_field]):
                try:
                    toes.append(t[np.where(t<=d)][-1])
                except:
                    toes.append(np.nan)

            base_dist_toes['toe']=toes


            print(f"Classifying beachfaces as going from {txt} shorelines to dune toes.")

            # preprocessing
            if mode=='gt':
                tmp_lookup=base_dist_toes.groupby(["location","raw_date","tr_id"])[toe_field,baseline_field].first().reset_index()
                tmp_merge=pd.merge(tmp_lookup,gdf_loc, how='right', on=["location","raw_date","tr_id"])
                tmp_merge=tmp_merge.dropna(subset=[baseline_field])[["distance",baseline_field,"location","raw_date","tr_id","toe","slope"]]
            else:

                tmp_lookup=base_dist_toes.groupby([*group_by_fields,"tr_id"])[toe_field,baseline_field].first().reset_index()
                tmp_merge=pd.merge(tmp_lookup,gdf_loc, how='right', on=["location","raw_date","tr_id"])
                tmp_merge=tmp_merge.dropna(subset=[baseline_field])[["distance",baseline_field,"location","raw_date","tr_id","toe","slope"]]

            # conditions
            tmp_merge['beachface']=['bf' if tmp_merge.loc[i,distance_field]>=tmp_merge.loc[i,toe_field] and tmp_merge.loc[i,distance_field] <= tmp_merge.loc[i,baseline_field]
             else "land" if tmp_merge.loc[i,distance_field]<tmp_merge.loc[i,toe_field]
             else "water" if tmp_merge.loc[i,distance_field]>tmp_merge.loc[i,toe_field]
             else np.nan for i in range(tmp_merge.shape[0])]



        else: # if we do not want to limit the correction to the beachface statistics, but we want to use the full transect stats

            merged_tmp=pd.merge(shores_to_corr,orig_shore_base_distances, on=[*group_by_fields], how='right',
                                suffixes=('_shore','_tr'))


        if isinstance(slope_value, str):

            if bool(limit_correction) == True:

                df_in=tmp_merge.query("beachface=='bf'")
            else:

                df_in=pd.merge(merged_tmp.astype({"tr_id": int, "raw_date": int, "location":str}),
                   gdf_loc.astype({"tr_id": int, "raw_date": int, "location":str})[['location','raw_date','tr_id','slope']],
                   how='right',on=['location','raw_date','tr_id'])


            ops_dict={"mean":np.nanmean,
                      "median":np.nanmedian,
                      "min":np.min,
                      "max":np.max,
                      "range":np.ptp,
                      "std":np.nanstd,
                      "var":np.nanvar
                     }


            stats=pd.DataFrame()
            for i in ops_dict.keys():

                tmp=pd.Series(df_in.groupby(["raw_date","tr_id"]).slope.apply(ops_dict[i]),
                          name=f"{i}_beachface_slope")
                stats=pd.concat([stats,tmp], axis=1)
                stats=stats.set_index(pd.MultiIndex.from_tuples(stats.index, names=("raw_date","tr_id")))
            stats["location"]=location

            slope_field=[stat for stat in stats.columns if stat.startswith(slope_value) ][0]

            if bool(limit_correction) == True:
                print(f"Using {slope_field} of beachfaces to correct shorelines.")
            else:
                print(f"Using full transect {slope_field} to correct shorelines.")

            stats=stats.reset_index()
            stats['raw_date']=[int(i)for i in stats.raw_date]


            orig_shore_trs_stats=pd.merge(orig_shore_base_distances,stats,on=["tr_id","location","raw_date"], how='left')

            if bool(replace_slope_outliers)==True:

                # create temp dataframes to store survey-level slope values stats and derive 3 sigmas thresholds per date
                survey_slope_field_stat=stats.groupby(["raw_date"])[slope_field].describe().reset_index()
                survey_slope_field_stat["sigmas_3"]=survey_slope_field_stat.loc[:,'50%']*3

                orig_shore_trs_stats=pd.merge(orig_shore_trs_stats,survey_slope_field_stat[['raw_date','sigmas_3','50%']],
                                  on=['raw_date'],how='left')

                # add the field 'outlier' and mark slope_values exceeding 3 std.
                orig_shore_trs_stats.loc[:,'outlier']=np.where(orig_shore_trs_stats[slope_field] > orig_shore_trs_stats['sigmas_3'],
                         True,False)

                # replace outliers with median survey values.
                orig_shore_trs_stats.loc[:,slope_field]=np.where(orig_shore_trs_stats[slope_field] > orig_shore_trs_stats['sigmas_3'],
                orig_shore_trs_stats['50%'],orig_shore_trs_stats[slope_field])
            else:
                pass


            correction_series=orig_shore_trs_stats.loc[:,slope_field]


        elif isinstance(slope_value,(int,float)):
            print(f"Using user-provided slope of {slope_value} to correct shorelines.")
            orig_shore_trs_stats = orig_shore_base_distances.assign(fixed_slope=slope_value)

            correction_series=orig_shore_trs_stats.fixed_slope


        # add tide height

        orig_shore_trs_stat_tide=pd.merge(orig_shore_trs_stats,
                                        shores_to_corr[[*group_by_fields,"tide_height"]],
                                        on=[*group_by_fields],how='left')

        orig_shore_trs_stat_tide["corr_dist"]=corr_baseline_distance(orig_shore_trs_stat_tide.sat_from_baseline,
                                                                        correction_series,
                                                                        orig_shore_trs_stat_tide.tide_height)

        orig_shore_trs_stat_tide["diff_corr"]=orig_shore_trs_stat_tide.sat_from_baseline - orig_shore_trs_stat_tide.corr_dist

        orig_shore_corr_dist_gdf=gpd.GeoDataFrame(orig_shore_trs_stat_tide, geometry="geometry", crs=crs)


    #_______ Apply threshold to correction, below this threshold, it is not corrected______________________________

        diffs=orig_shore_corr_dist_gdf.diff_corr.values
        corrs=orig_shore_corr_dist_gdf.corr_dist.values
        originals=orig_shore_corr_dist_gdf.sat_from_baseline.values


        # wether to correct or not shorelines closer to this threshold
        if baseline_threshold =='infer':
            baseline_threshold_value=np.round(threshold_multiotsu(orig_shore_corr_dist_gdf.diff_corr.dropna(), classes=2)[0],2)
            print(f"Inferred baseline_threshold of: {baseline_threshold_value} meters")

        elif isinstance(baseline_threshold, (int,float)):
            baseline_threshold_value=baseline_threshold
            print(f" Baseline distance threshold of: {baseline_threshold_value} meters")

        elif baseline_threshold == None:
            pass
        else:
            raise ValueError("Baseline_threshold must be either 'infer' or a numeric value.")

        updated_corr_dists=[c if d > baseline_threshold_value else o for d, c, o in zip(diffs,corrs,originals) ]
        orig_shore_corr_dist_gdf["updated_corr_dists"]=updated_corr_dists





        if 'slope_field' in locals():
            a=orig_shore_corr_dist_gdf.dropna(subset=[slope_field])[[*group_by_fields,"tr_id","sat_from_baseline","updated_corr_dists","geometry"]]
        else:
            a=orig_shore_corr_dist_gdf.dropna()[[*group_by_fields,"tr_id","sat_from_baseline","updated_corr_dists","geometry"]]
        a["updated_corr_dists"]=[a.iloc[i].sat_from_baseline if a.iloc[i].updated_corr_dists < 0 else a.iloc[i].updated_corr_dists for i in range(a.shape[0])]
        b=a.groupby([*group_by_fields])['tr_id'].apply(np.array).reset_index()
        b["updated_corr_dists"]=a.groupby([*group_by_fields])['updated_corr_dists'].apply(np.array).reset_index()['updated_corr_dists']
        b["tr_geometries"]=a.groupby([*group_by_fields])['geometry'].apply(np.array).reset_index()['geometry']



        # 5)_________________ Lines creation_________________________________________

        lines=[]

        for i, row in b.iterrows():
            groups=consecutive_ids(data = row.tr_id, limit=limit_vertex)

            if len(groups)>0:

                lines_group=[]
                for j,group in enumerate(groups):

                    trs=row.tr_geometries[group]
                    dis=row.updated_corr_dists[group]

                    line_pts=[]
                    for distance,transect in zip(dis,trs):
                        point=transect.interpolate(distance)
                        line_pts.append(point)

                    line=LineString(line_pts)
                    lines_group.append(line)

                if len(lines_group) >1:
                    lines_row=unary_union(lines_group)
                else:
                    lines_row=lines_group[0]

                lines.append(lines_row)
            else:
                print(f"WARNING:\n\
                {row.location}_{row.raw_date}_{row.water_index}_{row.thr_type} has {len(groups)} groups. Skipped.")
                lines.append(np.nan)


        lines=gpd.geoseries.GeoSeries(lines,crs=crs)
        b["corr_shore_geometry"]=lines.geometry
        final=pd.merge(shores_to_corr,b,on=[*group_by_fields], how='left')





        # 6)_________________ saving outputs________________________________________

        if bool(save_trs_details)==True:
            trs_details_file_name=f"trsdetails_{location}_{slope_value}_{limited}_{mode}.csv"
            trs_details_out_path=os.path.join(trs_details_folder,trs_details_file_name)

            orig_shore_corr_dist_gdf.to_csv(trs_details_out_path, index=False)
            print(f"File {trs_details_file_name} saving in {trs_details_folder}.")
        else:
            pass


        big_final=pd.concat([big_final,final], ignore_index=True)

        print(f"Done with {location}.")

    return big_final


def save_slope_corr_files(dsm_folder,baseline_folder,slope_profiles_folder, loc_codes,crs_dict_string,
                         shoreline,across_shore_resolution,alongshore_resolution,tick_length,side,
                          transects_gdf=None, exclude_loc=None, only_loc=None ):

    list_locations=list(set([getLoc(i,loc_codes) for i in glob.glob(f"{dsm_folder}/*.tif*")])) # get unique locations from list of DSMs

    if only_loc != None and exclude_loc != None:
        raise ValueError("Both only_loc and exclude_loc have been provided.\nSet both to None to work on all locations (default) or provide only one parameter to exclude or only work on specified locations.")
    else:
        pass

    if only_loc != None:
        list_locations=only_loc # list_locations is only those specified in only_loc parameter
        print(f"Working only on locations: {only_loc}")


    elif exclude_loc != None:
        list_locations=[loc for loc in list_locations if loc not in exclude_loc ]
        print(f"Excluded locations: {exclude_loc}")

        print("Locations to extract slope profiles from:\n")
        for loc in list_locations:
            print(f"{loc}")

    else:
        pass




    list_dsm=glob.glob(f"{dsm_folder}/*.tif*")


    # _______________Output Files preparations________________________________________



    if isinstance(across_shore_resolution,float):
        across_shore_resolution_txt=str(across_shore_resolution).replace(".","dot")
    else:
        across_shore_resolution_txt=str(across_shore_resolution)



        # Check if baselines CRS match rasters CRSs

    inconsistencies=pd.DataFrame()

    for location in list_locations:

        baseline=gpd.read_file(glob.glob(f"{baseline_folder}\{location}*")[0], geometry='geometry')

        imgs=glob.glob(f"{dsm_folder}\*{location}*")

        base_crs=baseline.crs["init"].split(":")[-1]
        raster_crs=[True if str(getCrs_from_raster_path(imgs[i])) != base_crs else False for i in range(len(imgs))]

        inconsis = [img for incons, img in zip(raster_crs, imgs) if incons]
        if len(inconsis) != 0:

            df_temp=pd.DataFrame({'location':location,
                                  'inconsis':inconsis,
                                 'raster_crs':[getCrs_from_raster_path(imgs[i]) for i in range(len(imgs))],
                                'baseline_crs':base_crs})
            inconsistencies=pd.concat([df_temp,inconsistencies], ignore_index=True)


    if inconsistencies.shape[0] !=0 :
        raise ValueError(f"CRS error. These rasters have different CRSs from their baseline CRS. Please reproject the baseline to match DSMs.")
        return inconsistencies
    else:
        pass

    # _______________SLOPE COMPUTATION________________________________________




    print("Preparing to compute slope rasters of all dates in the shorelines to correct.\n")
    for location in list_locations:

        gdf = pd.DataFrame()

        print(f"Working on {location} .")
        slope_file_name=f"slopeprofiles_{location}_{alongshore_resolution}_{tick_length}_{across_shore_resolution_txt}.csv"

        crs=crs_dict_string[location] # get crs of location

        baseline_location_path=[os.path.join(baseline_folder,baseline) for baseline in os.listdir(baseline_folder) if location in baseline ][0]
        baseline=gpd.read_file(baseline_location_path, crs=crs)

        list_dates=list(shoreline.query(f"location=='{location}'").raw_date.unique()) # get raw_dates of location
        locs=[]
        dates=[]
        for i in list_dsm:
            locs.append(getLoc(i, list_loc_codes=loc_codes))
            dates.append(getDate(i))
        datasets=pd.DataFrame({'loc':locs,
                     'date':dates})

        list_dates=[date_in for date_in in list_dates if date_in in datasets.query(f"loc=='{location}'").date.unique()]



    # 0) Creating the extraction transects if not provided

        if len(list_dates)==0:

            print(f"No DSMs found for {location}.")
            pass

        else:

            if isinstance(transects_gdf, gpd.GeoDataFrame):
                print(f"Transects GeoDataFrame provided.")
                transects=transects_gdf

            elif transects_gdf == None:
                transects=create_transects(baseline=baseline, side=side,
                                           sampling_step=alongshore_resolution, tick_length=tick_length,
                                           location=location, crs=crs)

            else:
                raise ValueError("Transects parameter is either None (default) or a GeoDataFrame.")



            for raw_date in list_dates:

                transects['raw_date']=raw_date

                # select the right dsm dataset
                dsm_path=[dsm for dsm in list_dsm if str(raw_date) == str(getDate(dsm)) and location == getLoc(dsm,list_loc_codes=loc_codes)]

                # double-check only one right DSM has been found
                if len(dsm_path) > 1:
                    raise ValueError(f"Multiple datasets for location: {location} and date: {raw_date} have been found.")
                elif len(dsm_path)==0:
                    raise ValueError(f"No datasets for location: {location} and date: {raw_date} have been found.")
                else:
                    print(f"Transects: location={location}; Date={raw_date} matched with DSM path= {dsm_path[0]}")

                # computing the slope raster
                print(f"Loading raster DSM into memory, for {location} at date: {raw_date}.")
                terr = rd.LoadGDAL(dsm_path[0])
                print(f"Computing its slope in degrees.")
                slope = rd.TerrainAttribute(terr, attrib='slope_degrees')
                del terr # to save memory
                print(f"Slope raster created.")

                # extracting slope

                tr_list = transects.tr_id.to_list()
                print(f"Extracting profiles.")
                for i in tr_list:
                    temp = get_profiles(dsm=dsm_path[0],
                                        transect_file=transects,
                                        transect_index=i,
                                        step=across_shore_resolution,
                                        location=location,
                                        date_string=raw_date,
                                        add_xy=True,
                                        add_terrain=slope)

                    temp.loc[:,"location"]=location
                    gdf = pd.concat([gdf, temp], ignore_index=True)



            # replacing DSM NoData (in Pix4D is - 10000.0) and 0.0 slopes with np.nan
            gdf["z"].replace(to_replace=-10000.0, value=np.nan, inplace=True)
            gdf["slope"].replace(to_replace=[0.0, -9999.0], value=[np.nan,np.nan], inplace=True)

            gdf["slope_tan_b"]=[tan(radians(slope_deg)) for slope_deg in gdf.slope]

            gdf["trs_along_space"]=alongshore_resolution
            gdf["trs_across_space"]=across_shore_resolution
            gdf["trs_length"]=tick_length
            gdf["side"]=side


            # Saving the data
            slope_profiles_out_path=os.path.join(slope_profiles_folder,slope_file_name)
            gdf.to_csv(slope_profiles_out_path,index=False)


        print(f"Correction profiles extraction finished and saved for {location}.\n")

        print("Ready to correct shorelines.")
    return gdf
