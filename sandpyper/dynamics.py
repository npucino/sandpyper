import numpy as np
from scipy.stats import median_abs_deviation, shapiro, normaltest
from statsmodels.api import qqplot
from tqdm.notebook import tqdm
import pandas as pd
import geopandas as gpd
import itertools
from itertools import product

from pysal.explore.giddy.markov import Markov
import matplotlib.pyplot as plt
import seaborn as sb
import glob

from sandpyper.outils import getListOfFiles, getLoc, create_spatial_id, spatial_id

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
    """It computes the r-BCDs at the transect level, based on the timeseries
    of elevation change magnituteds across the beachface dataset (markov_tag dataframe).

    Args:
        dataset (dataframe): Pandas dataframe with dh magnitude labelled.
        mode ("nnn","drop"): Insert "drop" to remove all cluster to no-cluster transitions,
        or 'nnn' (default) to keep them all and rename cluster-to-no clsuter state 'nnn'.
        unreal (str) : Insert "drop" (default) to remove transects that have less than the
        specified number of non-nnn points (thresh) or "keep" to keep them.
        thresh (int): Drop all rows with less than specified cluster transitions (i.e. non "nnn").
        min_points (int): Minumum of non-"nnn" points per transect to consider a transect reliable.
        field_markov_tags (str): The name of the column storing the magnitude classes. Default is "markov_tag".
        field_unique_id (str): The name of the column storing the unique ID of the points. Default is "geometry".
        field_discrete_time (str): he name of the column storing the period IDs. Default is "dt".

    Returns:
       A dataframes containing the steady-state distribution of each transect.
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
    else:
        pass

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

        path_trs=glob.glob(f"{dirNameTrans}\{loc}*")[0]
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
        arr_markov (array): Numpy array of markov transition matrix.
        weights_dict (dict): Dictionary with keys:dh classes, values: weigth (int). Especially useful for the e-BCDs magnitude trend (sign).
        store_neg: If True (default), use the subtraction for diminishing trends.

    Returns:
        BCD index, value of the trend, the sign ('-' or '+') for plotting purposes.
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
    From a dataframe containing the extracted points and a column specifying wether they are sand or non-sand, returns a multitemporal dataframe
    with time-periods sand-specific elevation changes.

    Args:
        date_field (str): the name of the column storing the survey date.
        sand_label_field (str): the name of the column storing the sand label (usually sand=0, no_sand=1).
        filter_classes (list): list of integers specifiying the label numbers of the sand_label_field that are sand. Default [0].
        common_field (str): name of the field where the points share the same name. It is usually the geometry or spatial IDs.

    Returns:
        A multitemporal dataframe of sand-specific elevation changes.
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

def sensitivity_tr_rbcd(df,
                       test_thresholds='max',
                       test_min_pts=[0,10,2]):

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
                      dirNameTrans=D.ProfileSet.dirNameTrans,
                      labels_order=D.tags_order,
                      loc_codes=D.ProfileSet.loc_codes,
                      crs_dict_string=D.ProfileSet.crs_dict_string)

                ss_transects_idx['thresh']=i[1]
                ss_transects_idx['min_pts']=i[0]

                ss_tr_big=pd.concat([ss_tr_big,ss_transects_idx], ignore_index=True)
            except:
                print("errore")

                pass

    return ss_tr_big

def plot_sensitivity_rbcds_transects(df, location, x_ticks=[0,2,4,6,8],figsize=(7,4),
                                     tr_xlims=(0,8), tr_ylims=(0,3), sign_ylims=(0,10)):


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
        ax.set_ylim(tr_ylims)
        ax.set_xlim(tr_xlims)


        plt.tight_layout()
        ax.get_legend().remove()
        ax2.get_legend().remove()


        plt.xticks(x_ticks)

        ax.set_title(f"pt: {minpts}")
        plt.tight_layout()
