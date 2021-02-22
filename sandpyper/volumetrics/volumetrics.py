import numpy as np
import os
import pandas as pd
from scipy.integrate import simps
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import dates
from matplotlib.dates import DateFormatter
from matplotlib.ticker import FixedFormatter,FixedLocator,AutoMinorLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import seaborn as sb


def getVol (dh):
    """ Return a volumetric change from altimetric change (dh)
    """
    return dh * 0.025


def prep_heatmap(df,lod,outliers=False,sigma_n=3):
    """
    Function to create a pivoted and filtered dataframe from dh_table of specific period-location combination (i.e. loc= pfa, dt= dt_3).
    Elevation differences within LoD (uncertain) can be set to zero and outliers can be eliminated.
    Each column is a transect and each row is a distance value along transect. Values are elevation differences.

    Warning:
        This function is to be used on a site-period specific slices of the dh_table.

    Args:
        df (Pandas dataframe): Location-period specific slice of dh_table.
        outliers: when True, use the specified number of standard deviation to exclude outliers. If False, retain all the points.
        sigma_n (int): number of standard deviation to use to exclude outliers (default=3).
        lod (path, value, False): if valid path to an LoD table, use the table.
            If a value is provided, use the value across all surveys. If False, do not apply LoD filter.
             All dh values within LoD will be set to zero.

    Returns:
        Pivoted and clean Pandas Dataframe, ready to be used to plot heatmaps and compute volumetrics.
    """


    # pivoting and sorting

    df["int_tr"]=df.tr_id.apply(lambda a : int(float(a)),convert_dtype=True) # add an int version of tr_id to sort better
    df.rename({'distance':'dist'},axis=1,inplace=True)
    df.dist.astype(float)
    df.sort_values(by=["int_tr","dt","dist"], inplace=True) # sort
    df_piv=df.pivot("dist","int_tr","dh")
    df_piv=df_piv.astype(float)

    df_piv.sort_index(axis=1, inplace=True) # sort columns, tr_id
    df_piv.sort_index(axis=0, ascending=False,inplace=True) # sort rows, distance


    if bool(outliers):
    # eliminating outliers

        thresh=sigma_n*np.nanstd(df_piv.astype(float).values) # find threshold for sigma outlier detection

        df_piv=df_piv[df_piv<thresh] # select only the values below the threshold
        df_piv=df_piv[-df_piv<thresh] # select only the negative values below the threshold

    else:
        pass

    if isinstance(lod, str):
        if os.path.isfile(lod):  # if a valid path, ope the table and use it

        # applying LoDs

            lod_table=pd.read_csv(lod)   # read in the lod table
            loc=df.location.iloc[0]
            dt=df.dt.iloc[0]
            lod=lod_table.query(f"location == '{loc}' and dt == '{dt}'").nmad # extract nmad
            cond=(df_piv >= -lod.values[0]) & (df_piv <=lod.values[0]) # create condition (within LoD) to mask the dataframe
            df_piv2=df_piv.mask(cond,0) # replace the values that satisfied the condition (within LoD) with zeroes
            df_piv2.set_index(df_piv2.index.astype(float), inplace=True)

            return df_piv2.sort_index(ascending=False)
        else:
            raise NameError ("Not a valid file or path.Please provide a valid path to the LoD table")

    elif isinstance(lod, (float,int)): # if a numeric, use this value across all surveys

        lod=float(lod)
        cond=(df_piv >= -lod) & (df_piv <=lod)
        df_piv2=df_piv.mask(cond,0)
        df_piv2.set_index(df_piv2.index.astype(float), inplace=True)

        return df_piv2.sort_index(ascending=False)

    else: # otherwise, use the default values, preset at global LoD 0.05.

        lod=float(lod_default)
        cond=(df_piv >= -lod) & (df_piv <=lod)
        df_piv2=df_piv.mask(cond,0)
        df_piv2.set_index(df_piv2.index.astype(float), inplace=True)

        return df_piv2.sort_index(ascending=False)


def fill_gaps(data_in,y_heat_bottom_limit,bottom=True,y_heat_start=0,spacing=0.1):
    """
    Function to fill the pivoted table (returned from prep_heatmap function) with missing across-shore distances, to align data on heatmaps.
    Empty rows (NaN) will be added on top (from 0 to the first valid distance) and, optionally on the bottom of each transect,
    (from the last valid distance to a specified seaward limit).

    Warning:
        This function assume along-transect distances to be going from land to water.

    Args:
        data_in (Pandas dataframe): Pivoted dataframe, where each column is a transect and row is a along-shore distance.
        y_heat_bottom_limit (int): Lower boundary distance to extend all transects.
        bottom (bool): If True (default), rows are extended seaward too, up to y_heat_bottom_limit. If False, only distances from 0 to the first valid values will be added.
        y_heat_start (int): Landward starting distance value (default=0)
        spacing(float): Sampling step (meters) used to extract points (default=0.1)
    Returns:
        Complete dataframe with extra rows of NaN added.
    """

    multiplier=0.1 * 100
    if bool(bottom)==True:
        bottom_fill_array=np.empty(((int(np.round(y_heat_bottom_limit + spacing - data_in.index[-1],1)*multiplier)),data_in.shape[1]))
        bottom_fill_array[:] = np.NaN
        to_concat_after=pd.DataFrame(data=bottom_fill_array,
                                     index=np.arange(data_in.index[-1],y_heat_bottom_limit + spacing,spacing),
                                     columns=data_in.columns)
    else:
        pass

    before_fill_array = np.empty((int(data_in.index[0]*multiplier),data_in.shape[1]))
    before_fill_array[:] = np.NaN
    to_concat_before=pd.DataFrame(data=before_fill_array,
                                  index=np.arange(y_heat_start,data_in.index[0],spacing),
                                  columns=data_in.columns)

    if bool(bottom)==True:
        return pd.concat([to_concat_before,data_in,to_concat_after.iloc[1:]])
    else:
        return pd.concat([to_concat_before,data_in])


def interpol_integrate(series):
    """
    Interpolate NaN values (non-sand) within the first and last valid points (from the swash to the landward end of each transect),
    and intergrate the area below this interoplated profile, to obtain transect specific volumetric change.

    Warning: for now, dx, interpolation and integration methods are fixed as it is used with Pandas method Series.apply(fn).
           Make dx a parameter.

    Args:
        Series (Pandas Series): series of elevation change with distance as indices.

    Returns:
        Volumetric change in cubic meters.
    """
    min_dist,max_dist=series.first_valid_index(),series.last_valid_index() # get distances of first and last sand points

    interpol=series.loc[min_dist:max_dist].interpolate() # interpolate linearly

    area_simps = simps(interpol.values, dx=0.1) # intergrate using Simpson's method

    return area_simps


def get_beachface_length(series):
    """
    Get across-shore beachface length from series of elevation change with distance as indices.

    Args:
        Series (Pandas Series): series of elevation change with distance as indices.
    Returns:
        Across-shore beachface length in meters.
    """

    min_dist,max_dist=series.first_valid_index(),series.last_valid_index()

    across_shore_beachface_length=np.round(max_dist - min_dist,1)

    return across_shore_beachface_length


def get_m3_m_location(data_in, transect_spacing=20):
    """
    Get alongshore-shore net volumetric change in cubic meters per meter of beach.

    Args:
        data_in (Pandas Dataframe): Dataframe generated by prep_heatmap function.
        transect_spacing (numeric): Spacing between transects in meters.
    Returns:
        Cubic meters of change per meter of beach alongshore, at the site level.
    """

    along_beach=data_in.shape[1]*transect_spacing # compute alongshore beachface length

    tot_vol=sum((data_in.apply(interpol_integrate, axis=0))*transect_spacing)  # compute net volume change


    return tot_vol/along_beach # return m3_m alongshore


def new_get_state_vol_table(sand_pts, lod, full_specs_table,
                        transect_spacing=20,outliers=False,sigma_n=3):
    """
    Function to compute location-level altimetric beach change statistics from the dh table.
    By default, only sand points beyond LoD are accounted for. Optionally, LoD filter can be turned off.
    The table contains info on:
    - monitoring period: location code, full name, period code, dates, number of days and valid points
    - absolute altimetric change: total beachface rising, lowering and net change
    - normalised altimetric change: meters of rising, lowering and net elevation change per valid survey point (MEC)


    Args:
        sand_pts (Pandas dataframe): dh_table.
        lod (path, value, False): if valid path to an LoD table, use the table.
            If a value is provided, use the value across all surveys. If False, do not apply LoD filter.
             All dh values within LoD will be set to zero.
        full_specs_table (False, path): Full path to the table with extended monitoring info. If False, monitoring period information are limited.
        transect_spacing (int): Alongshore spacing of transects (m)
        outliers (bool): when True, use the specified number of standard deviation to exclude outliers. If False, retain all the points.
        sigma_n (int): number of standard deviation to use to exclude outliers (default=3).


    Returns:
        A dataframe storing altimetric beach change info and other information for every period and location.
    """

    if isinstance(full_specs_table,bool):
        if bool(full_specs_table)== False:
            skip_details=True
        else:
            print("Please provide the path to the specs table.")
    elif os.path.isfile(full_specs_table):
        table_details=pd.read_csv(full_specs_table)
        skip_details=False
    else:
        raise TypeError("Not a valid path to the .csv file for the specs table.")

    tr_df_full=pd.DataFrame()

    locs=sand_pts.location.unique().tolist()

    for loc in locs:
        test_loc=sand_pts.query(f"location == '{loc}'")

        for dt in test_loc.dt.unique():
            test_dt=test_loc.query(f" dt =='{dt}'")
            data_in=prep_heatmap(test_dt,lod=lod,outliers=outliers,sigma_n=sigma_n)

            # compute erosion and deposition volumes at site level
            data_in_erosion=data_in[data_in < 0]
            data_in_deposition=data_in[data_in > 0]

            if bool(skip_details)==True:
                pass
            else:
                specs=table_details.query(f"loc_code=='{loc}' & dt=='{dt}'")
                full_loc=specs.loc_full.values[0]
                date_from=specs.date_from.values[0]
                date_to=specs.date_to.values[0]
                n_days=specs.n_days.values[0]

            beach_length=len(data_in.columns)*transect_spacing

            n_obs_valid=data_in.count().sum() # sand only, within beachface, LoD filtered (default)

            abs_in=data_in[data_in > 0].sum().sum() # total profiles rising
            abs_out=data_in[data_in < 0].sum().sum() # total profiles lowering
            abs_net_change=data_in.sum().sum() # net altimetric change

            mec_m=abs_net_change/beach_length # meters of elevation change per meter of beach

            norm_in=abs_in/n_obs_valid # meters of profile rising per valid point
            norm_out=abs_out/n_obs_valid # meters of profile lowering per valid point
            norm_net_change=abs_net_change/n_obs_valid # MEC


            tot_vol_depo=(data_in_deposition.apply(interpol_integrate, axis=0)*transect_spacing).sum()
            tot_vol_ero=(data_in_erosion.apply(interpol_integrate, axis=0)*transect_spacing).sum()
            net_vol_change= tot_vol_depo + tot_vol_ero
            location_m3_m=net_vol_change/beach_length

            if bool(skip_details)==False:
                df_dict={"location":loc,
                         "location_full":full_loc,
                        "dt": dt,
                         "date_from":date_from,
                         "date_to":date_to,
                         "n_days":n_days,

                         "abs_in":abs_in,
                         "abs_out":abs_out,
                         "abs_net_change":abs_net_change,
                         "mec_m":mec_m,

                         "norm_in":norm_in,
                         "norm_out":norm_out,
                         "norm_net_change":norm_net_change,

                         "tot_vol_depo":tot_vol_depo,
                         "tot_vol_ero":tot_vol_ero,
                         "net_vol_change": tot_vol_depo - abs(tot_vol_ero),
                         "location_m3_m": location_m3_m,

                         "n_obs_valid":n_obs_valid
                        }
            else:
                df_dict={"location":loc,
                            "dt": dt,

                             "abs_in":abs_in,
                             "abs_out":abs_out,
                             "abs_net_change":abs_net_change,
                             "mec_m":mec_m,

                             "norm_in":norm_in,
                             "norm_out":norm_out,
                             "norm_net_change":norm_net_change,

                             "tot_vol_depo":tot_vol_depo,
                             "tot_vol_ero":tot_vol_ero,
                             "net_vol_change": tot_vol_depo - abs(tot_vol_ero),
                             "location_m3_m": location_m3_m,

                             "n_obs_valid":n_obs_valid
                            }

            df=pd.DataFrame(df_dict,index=[0])

            tr_df_full=pd.concat([df,tr_df_full],ignore_index=True)

    return tr_df_full


def new_get_transects_vol_table(sand_pts, lod, full_specs_table,transect_spacing=20,   ### INTEGRATED
                        outliers=False,sigma_n=3):
    """
    Function to compute transect-level altimetric change statistics from the dh table.
    By default, only sand points beyond LoD are accounted for. Optionally, LoD filter can be turned off.
    The table contains info on:
    - monitoring period: location code, full name, period code, transect ID, dates, number of days and valid points
    - absolute altimetric change: total profile rising, lowering and net change
    - normalised altimetric change: meters of rising, lowering and net elevation change per valid survey point (MEC)

    Args:
        sand_pts (Pandas dataframe): dh_table.
        lod (path, value, False): if valid path to an LoD table, use the table.
            If a value is provided, use the value across all surveys. If False, do not apply LoD filter.
             All dh values within LoD will be set to zero.
        full_specs_table (False, path): Full path to the table with extended monitoring info. If False, monitoring period information are limited.
        lod (bool): when True (default), dh values within LoD are zeroed. If False, all the points are retained.
        outliers: when True, use the specified number of standard deviation to exclude outliers. If False, retain all the points.
        sigma_n (int): number of standard deviation to use to exclude outliers (default=3).

    Returns:
        A dataframe storing altimetric beach change info at the transect level, and other information for every period and location.
    """

    transects_df_full=pd.DataFrame()

    locs=sand_pts.location.unique().tolist()

    for loc in locs:

        test_loc=sand_pts.query(f"location == '{loc}'")

        for dt in test_loc.dt.unique():
            test_dt=test_loc.query(f" dt =='{dt}'")
            data_in=prep_heatmap(test_dt,lod,outliers=False,sigma_n=3)

            tr_ids=data_in.columns.values
            tr_ids.sort()

            if bool(full_specs_table):
                specs=full_specs_table.query(f"loc_code=='{loc}' & dt=='{dt}'")
                full_loc=specs.loc_full.values[0]
                date_from=specs.date_from.values[0]
                date_to=specs.date_to.values[0]
                n_days=specs.n_days.values[0]
            else:
                pass


            trs_volumes=data_in.apply(interpol_integrate, axis=0)
            trs_volumes.name="tot_vol_change"
            beach_lengths=data_in.apply(get_beachface_length, axis=0)


            tr_df=pd.DataFrame(trs_volumes)
            tr_df['m3_m']=(trs_volumes.values*transect_spacing)/beach_lengths.values # normalise the volume change computed in the transect by its cross-shore length
            tr_df=tr_df.reset_index()
            tr_df.rename({'int_tr':'tr_id'},axis=1, inplace=True)


            tr_df["n_obs_valid"]=data_in.count().values
            tr_df["abs_in"]=data_in[data_in > 0].sum().values
            tr_df["abs_out"]=data_in[data_in < 0].sum().values
            tr_df["abs_net_change"]=data_in.sum().values


            tr_df["mec_m"]=tr_df.abs_net_change.values/beach_lengths.values

            tr_df["norm_in"]=tr_df.abs_in.values/tr_df.n_obs_valid.values
            tr_df["norm_out"]=tr_df.abs_out.values/tr_df.n_obs_valid.values
            tr_df["norm_net_change"]=tr_df.abs_net_change.values/tr_df.n_obs_valid.values

            tr_df["dt"]=dt
            tr_df["location"]=loc

            if bool(full_specs_table):
                tr_df["location_full"]=full_loc
                tr_df["date_from"]=date_from
                tr_df["date_to"]=date_to
                tr_df["n_days"]=n_days

            else:
                pass

            transects_df_full=pd.concat([tr_df,transects_df_full],ignore_index=True)

    return transects_df_full


def new_plot_alongshore_change (sand_pts,mode,lod,full_specs_table,return_data=False,
                            location_subset=['wbl'],dt_subset=['dt_0'],ax2_y_lims=(-8,5),
                            save=False,save_path="C:\\jupyter\\images_ch_4\\volumetric_dynamics\\",dpi=300,img_type=".png",
                            from_land=True,from_origin=True,add_orient=False,
                            fig_size=(7.3,3),font_scale=1,plots_spacing=0,
                            bottom=False,y_heat_bottom_limit=80,transect_spacing=20
                           ,outliers=False,sigma_n=3,):
    """
    Display and optionally save alongshore altimetric and volumetric beach changes plots.
    A subset of locations and periods can be plotted.
    If LoD parameter is True (default), then white cells in the altimetric heatmap are values within LoD. Grey cells is no data or no sand points.
    Optionally, LoD filter can be turned off.

    Args:
        sand_pts (Pandas dataframe): dh_table.
        mode (str): if 'subset', only a subset of locations and dts are plotted. If 'all', all periods and locations are plotted. .
        lod (path, value, False): if valid path to an LoD table, use the table.
        If a value is provided, use the value across all surveys. If False, do not apply LoD filter. All dh values within LoD will be set to zero.
        full_specs_table (False, path): Full path to the table with extended monitoring info. If False, monitoring period information are limited.
        location_subset (list): list of strings containing the location codes (e.g. wbl) to be plotted.
        dt_subset (list): list of strings containing the period codes (e.g. dt_0) to be plotted.
        ax2_y_lims (tuple): limits of y-axis of alonghsore volumetric change plot. Default is (-8,5).
        save (bool): If True, saves the plots in the specified save_path. False is default.
        save_path (path): Full path to a folder (e.g. C:\\jupyter\\images\\) where to save plots.
        dpi (int): Resolution in Dot Per Inch (DPI) to save the images.
        img_type (str): '.png','.pdf', '.ps', '.svg'. Format of the saved figures.
        from_land (bool): If True (default), cross-shore distances are transformed into landward distances, where 0 is the end of beachface.
        from_origin (bool): If True (default), transect IDS are transformed in alongshore distance from origin (tr_id=0). It requires regularly spaced transects.
        add_orient (bool): if True, an additional lineplot is added to the volumetric plot containing orientation info. It needs pre-computed orientations (tr_orient parameter) (TO UPDAte). False is default.
        tr_orient (Pandas dataframe): dataframe containign transect orientations.
        fig_size (tuple): Tuple of float to specify images size. Default is (7.3,3).
        font_scale (float): Scale of text. Default=1.
        plots_spacing (flaot): Vertical spacing of the heatmap and alongshore change plots. Default = 0.
        bottom (bool): bottom (bool): If True (default), rows are extended seaward too, up to y_heat_bottom_limit. If False, only distances from 0 to the first valid values will be added.
        y_heat_bottom_limit(int): y_heat_bottom_limit (int): Lower boundary distance (seaward) to extend all transects to.
        transect_spacing(int): Alongshore spacing of transects (m).
        outliers (bool): when True, use the specified number of standard deviation to exclude outliers. If False, retain all the points.
        sigma_n (int): number of standard deviation to use to exclude outliers (default=3).

    Returns:
        Prints and save alongshore beach change plots.
    """

    sb.set_context("paper", font_scale=font_scale, rc={"lines.linewidth": 0.8})

    if isinstance(full_specs_table,bool):
        if bool(full_specs_table)== False:
            skip_details=True
        else:
            print("Please provide the path to the specs table.")
    elif os.path.isfile(full_specs_table):
        table_details=pd.read_csv(full_specs_table)
        skip_details=False
    else:
        raise TypeError("Not a valid path to the .csv file for the specs table.")

    land_limits=pd.DataFrame(sand_pts.groupby(["location"]).distance.max()).reset_index()

    locations_to_analyse=sand_pts.location.unique()

    if bool(from_origin)==True:
        xlabel=''
    else:
        xlabel='Transect ID'

    if mode == "subset":
        locations_to_analyse=location_subset
        dt_to_analyse=dt_subset

    elif mode =="all":
        locations_to_analyse=sand_pts.location.unique()
        dt_to_analyse=sand_pts.query(f"location == '{loc}'").dt.unique()

    for loc in locations_to_analyse:
        temp_loc=sand_pts.query(f"location == '{loc}'")

        for dt in dt_to_analyse:


            # subset the data
            temp=sand_pts.query(f"location == '{loc}' and dt =='{dt}'")
            if bool(add_orient)==True:
                tr_or=tr_orient.query(f"location == '{loc}'")
            else:
                pass

            # prepare axes and figure
            f, (ax,ax2) = plt.subplots(nrows=2,figsize=fig_size,sharex=True,
                                       gridspec_kw={'hspace':plots_spacing})


            #_____________data_preparation______________

            # compute beach length based on number of transects
            beach_length=len(temp.tr_id.unique()) * transect_spacing

            # prepare the data to be suitable for the heatmap
            data_in=prep_heatmap(temp,lod=lod,outliers=outliers,sigma_n=sigma_n)



            if bool(skip_details)==True:
                full_loc=loc
                lod=0.05
            else:

                specs=table_details.query(f"loc_code=='{loc}' & dt=='{dt}'")
                full_loc=specs.loc_full.values[0]
                date_from=specs.date_from.values[0]
                date_to=specs.date_to.values[0]
                n_days=specs.n_days.values[0]



            if isinstance(lod, str):
                if os.path.isfile(lod):
                    table=pd.read_csv(lod_table_path)
                    lod=np.round(lod_table.query(f"location == '{loc}' and dt == '{dt}'").nmad.values[0],2) # extract nmad
            elif isinstance(lod, (float,int)):
                lod=lod
            else: lod=0.05

    # FIGURE_______________________________________________________________________

            #
            axins = ax.inset_axes(bounds=[1.02, 0,0.04,1])

            if bool(from_land)==True:

                land_lim=land_limits.query(f"location=='{loc}'").distance.values[0]
                data_in["m_from_land"]=np.round(land_lim - data_in.index,1)
                data_in.set_index("m_from_land", inplace=True)
            else:
                pass

            if bool(from_origin)==True:
                data_in.columns=data_in.columns.astype("int")*transect_spacing
            else:
                pass

            data_in_filled=fill_gaps(data_in,y_heat_bottom_limit,bottom=bottom)
            print(f"Working on {loc} at {dt}")

        #_______AX__________________________________________________________________

            sb.heatmap(data=data_in_filled, yticklabels=50, xticklabels=10,facecolor='w',robust=True, center=0, ax=ax,
                       cbar_kws={'label': u'Δh m AHD'}, cbar=True , cbar_ax=axins,
                       cmap="seismic_r",vmin=-0.8,vmax=0.8 )


            #_________________________________BACKGROUND COLOR AND TRANSPARENCY_____
            ax.patch.set_facecolor('grey')
            ax.patch.set_alpha(0.2)

            #_________________________________AXIS LABELS_____
            ax.set_xlabel("")
            ax.set_ylabel("Cross. distance (m)")
            ax.set_title(f'')


            #__________________________________SPINES,TICKS_AND_GRIDS_________________________________
            ax.get_xaxis().set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)
            ax.spines['right'].set_visible(True)
            ax.spines['top'].set_visible(True)

            ax.grid(b=True,which='minor',linewidth=0.3, ls='-')
            ax.grid(b=True,which='major',linewidth=0.8,ls='-')



            ax.set_xticks(ax.get_xticks() - 0.5)

            tmp_list_along_dists=np.arange(0,y_heat_bottom_limit+10,10)
            y_formatter_list_values=tmp_list_along_dists.astype("str").tolist()
            y_formatter_list_values[-1]=""
            y_formatter_list_locators=[value * 10 for value in tmp_list_along_dists]
            y_formatter_list_locators,y_formatter_list_values

            y_formatter = FixedFormatter(y_formatter_list_values)
            y_locator = FixedLocator(y_formatter_list_locators)

            ax.yaxis.set_major_formatter(y_formatter)
            ax.yaxis.set_major_locator(y_locator)
            ax.yaxis.set_minor_locator(AutoMinorLocator(4))
            ax.grid(which='minor',axis='y')

            ax.set_ylim(y_heat_bottom_limit*10,0)
            ax.tick_params(axis=u'x', which=u'both',length=0)


        #__AX2___________________________________________________________________

            red_patch = mpatches.Patch(color='orange', label='Erosion')
            blue_patch = mpatches.Patch(color='skyblue', label='Deposition')


            trs_volumes=data_in.apply(interpol_integrate, axis=0)
            beach_lengths=data_in.apply(get_beachface_length, axis=0)

            m3_m=(trs_volumes*transect_spacing)/beach_lengths

            trs_volumes.reset_index().rename({'int_tr':'tr_id'},axis=1, inplace=True)
            trs_volumes.name="dh"
            tr_df=pd.DataFrame(trs_volumes)

            tr_df.reset_index(inplace=True)
            tr_df["net_volume_change_m3"]=tr_df.dh
            tr_df["net_balance_m3_m"]=m3_m.values

            tr_df.set_index("int_tr")
            tr_df.reset_index(inplace=True)

            sb.lineplot(data=tr_df,x='index', y='net_balance_m3_m', ax=ax2, color='k')
            ax2.set_ylim(ax2_y_lims)
            ax2.yaxis.set_minor_locator(AutoMinorLocator(4))

            #ax3________________________________________

            if bool(add_orient)==True:
                ax3 = ax2.twinx()
                ax3.set_ylabel('Transect Orientation (° TN)')


                if bool(from_origin)==True:
                    ax3.scatter(tr_or.tr_id,tr_or.tr_orientation, alpha=0.5, c="k")
                else:
                    ax3.scatter(tr_or.tr_id,tr_or.tr_orientation, alpha=0.5, c="k")
            else:
                pass


            #FILLS________________________________________

            line_x,line_y=ax2.lines[0].get_data()

            ax2.fill_between(line_x, line_y, where=(line_y>0),
                            interpolate=True, color='skyblue',alpha=0.3)

            ax2.fill_between(line_x, line_y, where=(line_y<0),
                            interpolate=True, color='orange', alpha=0.3)


            ax2.grid(b=True,which='minor',linewidth=0.5, ls='-')
            ax2.grid(b=True,which='major',linewidth=0.8,ls='-')

            ax2.set_xlabel("Along. distance (m)")
            ax2.set_ylabel(u'Net Δv (m³/m)')
            ax2.axhline(0, color='k', ls="--",zorder=1)
            ax2.set_title(f'')

            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(True)
            ax2.spines['left'].set_visible(True)
            ax2.spines['bottom'].set_visible(True)
            ax2.grid(which='minor',axis='y')

            if bool(skip_details)==False:
                date_from_str=pd.to_datetime(str(date_from)).strftime("%d %b '%y")
                date_to_str=pd.to_datetime(str(date_to)).strftime("%d %b '%y")

                f.suptitle(f"Beachface change in {full_loc} from {date_from_str} to {date_to_str} (LoD = {lod} m)")
            else:

                f.suptitle(f"Beachface change in {loc} for {dt} (LoD = {lod} m)")

            axs=f.get_axes()

            if bool(add_orient)==True:
                f.align_ylabels(axs[:-2])
            else:
                f.align_ylabels(axs[:-1])


            ax.set_zorder(1)
            ax2.set_zorder(0)

            if bool(save)==True:
                savetxt=save_path+loc+'_'+dt+"_AlongChange"+img_type
                f.savefig(savetxt,dpi=dpi,bbox_inches="tight")
            else:
                pass
            plt.show()

    if return_data==True:
        return data_in_filled


def plot_mec_evolution(volumetrics,location_field, date_format="%d.%m.%y", scale_mode="equal",
                       dates_step=50, x_limits=(-0.41,0.41), x_binning=5, figure_size=(7, 4), font_scale=0.75,
                       sort_locations=True, loc_order=["pfa","wbl","mar","apo","prd","leo","por","cow","inv","sea"],
                       dpi=300, img_type=".png", save_fig=False, name_fig=f"Normalised Dynamics in Victoria_bigfont",
                       save_path='C:\\jupyter\\images_ch_4\\normalised_changes\\') :
    """
    Display and optionally save global volumetric timeseries plots, displaying period-specific Mean Elevation Change (mec, in m) and cumulative mec (since start of the monitoring), across lcoations.

    Args:
        volumetrics (Pandas dataframe): location-level-volumetrics table obtained from get_state_vol_table function.
        date_format (str): format to plot dates on y-axis. Default="%d.%m.%y".
        scale_mode (str) ("optimised","equal"): If "equal" (Default), all locations subplots will have the same x-axis ticks spacing (reccomended for comaprison purposes).
        If "optimised", an attempt to estimate the best tick steps to use by analysing absolute range of mec values.
        dates_step (int): Frequency of days between each tick in the y-axis (dates). Default=50.
        x_limits (tuple of floats): tuple containing min and max values of mec. Used to get unifor subplots widths.
        Note: currently, an exeption is hard-coded for Inverloch, which is one ad-hoc case used during code development. (TO BE UPDated)
        x_binning (int):
        figure_size (tuple): Tuple of float to specify images size. Default is (7.4).
        font_scale (float): Scale of text.
        sort_locations (bool): Wether or not to sort the locations according to the loc-order provided.
        loc_order (list): Location order to plot locations.
        dpi (int): Resolution in Dot Per Inch (DPI) to save the images.
        img_type (str):  '.png','.pdf', '.ps', '.svg'. Format of the saved figures.
        save_fig (bool): If True, saves the plots in the specified save_path. False is default.
        name_fig (str): Name of the figure file to be saved.
        save_path(str): Full path to a folder (e.g. C:\\jupyter\\images\\) where to save plots.

    Returns:
        Prints and optionally save global period-specific and cumulative MEC timeseries across all lcoations.
    """


    pd.options.mode.chained_assignment = None  # default='warn'


    sb.set_context('paper', font_scale=font_scale)
    myFmt=DateFormatter("%d.%m.%y")


    # sort the locations
    if bool(sort_locations)==True:

        sorterIndex = dict(zip(loc_order,range(len(loc_order))))
        volumetrics['loc_rank']=volumetrics['location'].map(sorterIndex)
        volumetrics.sort_values(["loc_rank","dt"], inplace=True)

    else:
        pass

    # replace long location names with shorter versions

    volumetrics.replace("Point Roadknight","Pt. Roadk.", inplace=True)
    volumetrics.replace("St. Leonards","St. Leo.", inplace=True)

    num_subplots=volumetrics.location.unique().shape[0]
    if num_subplots > 1:

        fig, axs = plt.subplots(1, num_subplots,
                            sharey=True, sharex=False,
                            squeeze=True,
                            figsize=figure_size)

        for ax_i,loc in zip(axs.flatten(),volumetrics.location.unique()):
        #     print(ax,loc)

            data_in=volumetrics.query(f"location=='{loc}'")
            data_in.sort_values("date_to",inplace=True)
            data_in["cum_change"]=data_in.norm_net_change.cumsum() # add the cumulative change curve

            first_date_from=data_in.date_from.iloc[0]

            full_loc=data_in.loc[:,location_field].iloc[0]
            x=data_in.norm_net_change
            x2=data_in.cum_change
            y=dates.date2num(data_in.date_to)
            y_scatter=dates.date2num(data_in.date_from)

            # LinePlots

            # normalised change
            ax_i.plot(x,y, lw=0.8, c="k")

            # cumulative norm.change
            ax_i.plot(x2,y, lw=0.8, ls="--", c="k", zorder=4)

            # from start of monitoring to first "date_to"
            x_start=np.array([0,x.iloc[0]])
            y_start=[y_scatter[0],y[0]]

            ax_i.plot(x_start,y_start, lw=1, c="k",zorder=3)

            # Vertical Line
            ax_i.axvline(0, color='grey', ls='-', lw=0.8, zorder=0)

            # Spines
            ax_i.spines['right'].set_visible(False)
            ax_i.spines['left'].set_visible(False)
            ax_i.spines['top'].set_visible(False)
            ax_i.spines['bottom'].set_linewidth(0.8)
            ax_i.spines['bottom'].set_color("k")

            # Grid
            ax_i.grid(color='grey', linestyle='-', linewidth=0.8, alpha=0.3)

            # ScatterPlots
            ax_i.scatter(x,y, c="k", s=5,zorder=5)
            ax_i.scatter(0,first_date_from,c="k",s=5,zorder=5)

            # ticks
            if scale_mode=="optimised":

                abs_range=abs(max(x) - (min(x)))

                if abs_range < 0.2:
                    tick_step=0.05

                else:
                    tick_step=0.1


                ticks_value=np.arange(round_special(min(x),0.05),round_special(max(x),0.05)+tick_step, tick_step)
                ax_i.set_xticks(ticks_value)

            elif scale_mode=="equal":             # Inverloch different scale, for display purposes
                if loc != 'inv':

                    ax_i.set_xlim(x_limits)
                else:
                    ax_i.set_xlim(-1.1,1.1)

            else:
                pass

            start, end = ax_i.get_ylim()
            ax_i.yaxis.set_ticks(np.arange(start, end, dates_step))

            # SubPlot Title
            ax_i.set_title(f'{full_loc}')
            ax_i.yaxis.grid(True)
            ax_i.yaxis.set_major_formatter(myFmt)


        for ax in fig.axes:
            plt.sca(ax)
            plt.locator_params(axis='x', nbins=x_binning)

            plt.xticks(rotation=90)

    else:

        fig, ax = plt.subplots(figsize=figure_size)


    #     print(ax,loc)

        data_in=volumetrics
        data_in.sort_values("date_to",inplace=True)
        data_in["cum_change"]=data_in.norm_net_change.cumsum() # add the cumulative change curve

        first_date_from=data_in.date_from.iloc[0]

        full_loc=data_in.loc[:,location_field].iloc[0]
        x=data_in.norm_net_change
        x2=data_in.cum_change
        y=dates.date2num(data_in.date_to)
        y_scatter=dates.date2num(data_in.date_from)

        # LinePlots

        # normalised change
        ax.plot(x,y, lw=0.8, c="k")

        # cumulative norm.change
        ax.plot(x2,y, lw=0.8, ls="--", c="k", zorder=4)

        # from start of monitoring to first "date_to"
        x_start=np.array([0,x.iloc[0]])
        y_start=[y_scatter[0],y[0]]

        ax.plot(x_start,y_start, lw=1, c="k",zorder=3)

        # Vertical Line
        ax.axvline(0, color='grey', ls='-', lw=0.8, zorder=0)

        # Spines
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_linewidth(0.8)
        ax.spines['bottom'].set_color("k")

        # Grid
        ax.grid(color='grey', linestyle='-', linewidth=0.8, alpha=0.3)

        # ScatterPlots
        ax.scatter(x,y, c="k", s=5,zorder=5)
        ax.scatter(0,first_date_from,c="k",s=5,zorder=5)

        # ticks
        if scale_mode=="optimised":

            abs_range=abs(max(x) - (min(x)))

            if abs_range < 0.2:
                tick_step=0.05

            else:
                tick_step=0.1


            ticks_value=np.arange(round_special(min(x),0.05),round_special(max(x),0.05)+tick_step, tick_step)
            ax.set_xticks(ticks_value)

        elif scale_mode=="equal":             # Inverloch different scale, for display purposes
            if loc != 'inv':

                ax.set_xlim(x_limits)
            else:
                ax.set_xlim(-1.1,1.1)

        else:
            pass

        start, end = ax.get_ylim()
        ax.yaxis.set_ticks(np.arange(start, end, dates_step))

        # SubPlot Title
        ax.set_title(f'{full_loc}')
        ax.yaxis.grid(True)
        ax.yaxis.set_major_formatter(myFmt)


    for ax in fig.axes:
        plt.sca(ax)
        plt.locator_params(axis='x', nbins=x_binning)

        plt.xticks(rotation=90)




    fig.tight_layout(pad=0.2)

    if bool(save_fig)==True:

        savetxt= save_path + name_fig + img_type

        plt.savefig(savetxt, dpi=dpi)
    else:
        pass
