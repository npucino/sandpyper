import numpy as np
from scipy.stats import median_abs_deviation, shapiro, normaltest
from statsmodels.api import qqplot
from tqdm.notebook import tqdm
import pandas as pd
import geopandas as gpd
import itertools
from itertools import product, combinations

from pysal.explore.giddy.markov import Markov
import matplotlib.pyplot as plt
import seaborn as sb

from sandpyper.outils import getListOfFiles, getLoc, create_spatial_id

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
            abs_data=abs(data_lod)

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
            abs_data=abs(data_lod)

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


def attach_trs_geometry(markov_transects_df, dirNameTrans, list_loc_codes):
    """Attach transect geometries to the transect-specific BCDs dataframe.
    Args:
        markov_transects_df (Pandas dataframe): Dataframe storing BCDs at the transect level.
        dirNameTrans (str): Full path to the folder where the transects are stored.
        list_loc_codes (list): list of strings containing location codes.

    Returns:
        Dataframe with the geometries attached, ready to be mapped.
    """
    return_df = pd.DataFrame()

    list_trans = getListOfFiles(dirNameTrans)
    for i in list_trans:

        transect_in = gpd.read_file(i)
        transect_in.rename({"TR_ID": "tr_id"}, axis=1, inplace=True)
        loc = getLoc(i, list_loc_codes)

        sub_markovs_trs = markov_transects_df.query(f"location=='{loc}'")
        sub_markovs_trs["geometry"] = pd.merge(
            sub_markovs_trs, transect_in, how="left", on="tr_id"
        )["geometry"].values

        return_df = pd.concat([return_df, sub_markovs_trs], ignore_index=True)
    return return_df


def infer_weights(data, markov_tag_field="markov_tag"):
    """Compute weights from dataset with markov labels to use for e-BCDs computation.
        The medians of each magnitude class will be used as weight.

    Args:
        data (Pandas Dataframe): Pandas dataframe.
        markov_tag_field (str): Name of the column where markov tags are stored.

    Returns:
        dict, containing the markov tags and associated weights.
    """

    dict_medians = {}

    mags = []
    for i in range(data.shape[0]):

        letter = data.loc[i, markov_tag_field][0]
        if letter == "u":
            mag = "undefined"
        elif letter == "s":
            mag = "slight"
        elif letter == "m":
            mag = "medium"
        elif letter == "h":
            mag = "high"
        elif letter == "e":
            mag = "extreme"
        mags.append(mag)

    data["magnitude_class"] = mags

    # create a dictionary with all the medians of the classes
    dict_medians = {}
    for classe in data.magnitude_class.unique():
        class_in = data.query(f"magnitude_class=='{classe}'")
        class_median = np.median(np.abs(class_in.dh))
        if classe == "undefined":

            dict_medians.update(
                {"ue": np.round(class_median, 2), "ud": np.round(class_median, 2)}
            )
        elif classe == "slight":

            dict_medians.update(
                {"se": np.round(class_median, 2), "sd": np.round(class_median, 2)}
            )

        elif classe == "medium":

            dict_medians.update(
                {"me": np.round(class_median, 2), "md": np.round(class_median, 2)}
            )
        elif classe == "high":

            dict_medians.update(
                {"he": np.round(class_median, 2), "hd": np.round(class_median, 2)}
            )
        elif classe == "extreme":

            dict_medians.update(
                {"ee": np.round(class_median, 2), "ed": np.round(class_median, 2)}
            )

    return dict_medians


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
                weigth_adhoc = state_1_w * state_2_w
                weigth_adhoc_trend = state_1_w * (-(state_2_w))

            else:
                weigth_adhoc = state_1_w * ((state_2_w))

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


def BCDs_compute(
    dataset,
    weights_dict,
    mode,
    unique_field="geometry",
    label_order=["ed", "hd", "md", "sd", "ud", "nnn", "ue", "se", "me", "he", "ee"],
    store_neg=True,
    plot_it=False,
    fig_size=(6, 4),
    heat_annot_size=10,
    font_scale=0.75,
    dpi=300,
    save_it=False,
    save_output="C:\\your\\preferred\\folder\\",
):

    """It computes all the first order stochastic transition matrices, based on the timeseries
    of elevation change magnituteds across the beachface dataset (markov_tag dataframe), at the site level.

    Warning: changing label order is not supported as submatrix partitioning is hard-coded.

    Args:
        dataset (dataframe): Dataframe with dh values timeseries.
        weigth_dict (dict): dictionary containing each magnitude class as keys and value to be used to weigth each class as values.
        This can be manually set or objectively returned by the infer_weights function (reccommended).
        mode (float,"all","drop"): insert a float (default is 0.75) to indicate the percentage of time that
        the points need to be significant clusters across the periods. The no cluster state is renamed "nnn".
        Insert "drop" to remove all cluster to no-cluster transitions, or 'all' to keep them all and rename
        cluster-to-no clsuter state 'nnn'.
        unique_field (str) : The field contianing unique spatial IDs. Default is "geometry".
        label_order: order to arrange the states in the first-order and steady-state matrices.
        store_neg (bool): If True (default), use the subtraction for diminishing trends. Default True.
        plot_it (bool): Wether or not to plot the resulting first order matrices as heatmaps.
        fig_size (tuple): Size of Figures in inches. Default (6,4)
        font_scale (float): Scale of text.
        dpi (int): dpi used to produce the image.
        save_it (bool): Wether or not save the figures in the save_output parameter.
        save_output (str): DIR path to store figures if plot_it is True

    Returns:
       Two dataframes. One is the e-BCDs and the second is the steady-state distribution dataframes.
       Optionally plots the transition matrices and save them in the specified output folder.
    """

    sb.set_context(font_scale=font_scale)

    steady_state_victoria = pd.DataFrame()
    markov_indexes_victoria = pd.DataFrame()

    for loc in dataset.location.unique():

        dataset_piv = dataset.query(f"location=='{loc}'").pivot(
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

        n = dataset_piv.shape[0]
        t = dataset_piv.shape[1]

        arr = np.array(dataset_piv)

        m = Markov(arr)
        n_transitions = m.transitions.sum().sum()

        steady_state = pd.DataFrame(m.steady_state, index=m.classes, columns=[loc])
        steady_state_victoria = pd.concat([steady_state, steady_state_victoria], axis=1)

        markov_df = pd.DataFrame(np.round(m.p, 3), index=m.classes, columns=m.classes)

        # if any column is missing from label_order, then add both row and column
        # and populate with zeroes
        if markov_df.columns.all != 11:  # must be 11, 5 for process plus nnn
            # which one is missing?
            missing_states = [
                state for state in label_order if state not in markov_df.columns
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

        markov_ordered = markov_df.loc[label_order, label_order]

        # When no transition occurred, replace NaN with a 0.
        markov_ordered.fillna(0, inplace=True)

        # reverted orders for extremes, to increase readibility
        dd = markov_ordered.iloc[:5, :5]
        dd = dd[dd.columns[::-1]]

        ee = markov_ordered.iloc[6:, 6:]
        ee = ee.reindex(index=ee.index[::-1])
        de = markov_ordered.iloc[:5, 6:]

        ed = markov_ordered.iloc[6:, :5]
        ed = ed.reindex(index=ed.index[::-1])
        ed = ed[ed.columns[::-1]]

        list_markovs = [ee, ed, dd, de]
        dict_markovs = {"ee": ee, "ed": ed, "dd": dd, "de": de}

        # create index dataframe

        for arr_markov in dict_markovs.keys():

            idx, trend, sign = get_coastal_Markov(
                dict_markovs[arr_markov], weights_dict=weights_dict, store_neg=store_neg
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

        if bool(plot_it):

            titles = ["Erosional", "Recovery", "Depositional", "Vulnerability"]
            cmaps = ["Reds", "Greens", "Blues", "PuRd"]

            std_excluding_nnn = (
                markov_ordered.loc[
                    markov_ordered.index != "nnn", markov_ordered.columns != "nnn"
                ]
                .values.flatten()
                .std()
            )
            exclude_outliers = np.round(3 * std_excluding_nnn, 1)

            f2, axs = plt.subplots(nrows=2, ncols=2, figsize=fig_size)

            for ax_i, heat, title, cmap_i in zip(
                axs.flatten(), list_markovs, titles, cmaps
            ):
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
                title = f2.suptitle(f"{loc} (n={n},t={t}, trans:{int(n_transitions)}) ")
                title.set_position([0.5, 1.03])
                f2.tight_layout(pad=1)

            if bool(save_it):
                f2.savefig(f"{save_output}{loc}_sub_matrices_.png", dpi=dpi)
            else:
                pass
        else:
            pass

    return markov_indexes_victoria, steady_state_victoria


def steady_state_transect(
    dataset,
    mode="nnn",
    unreal="drop",
    thresh=8,
    min_points=20,
    field_markov_tags="markov_tag",
    field_unique_id="geometry",
    field_discrete_time="dt",
    use_neg=True,
):
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
        use_neg (bool): If True (default), use the subtraction for diminishing trends. Default True.

    Returns:
       A dataframes containing the steady-state distribution of each transect.
    """

    steady_state_tr = pd.DataFrame()
    markov_indexes_tr = pd.DataFrame()

    # in mode, insert "drop" to remove cluster to no-cluster transitions or "nnn" to retain them and label them "nnn"
    # If drop, only cluster to cluster points that happend during all the
    # surveys are retained

    for loc in tqdm(dataset.location.unique()):
        data_loc = dataset.query(f"location=='{loc}'")

        unreliable_trs = []

        for tr_id in data_loc.tr_id.unique():

            data_tr = data_loc.query(f"tr_id=='{tr_id}'")

            if data_tr.empty:
                data_tr = data_loc.query(f"tr_id=={tr_id}")

            data_piv = data_tr.pivot(
                values=field_markov_tags,
                index=field_unique_id,
                columns=field_discrete_time,
            )

            if mode == "nnn":
                # drop all rows with less than specified cluster transitions
                data_piv.dropna(axis=0, thresh=thresh, inplace=True)
                # all the remaining NaN will be named 'nnn'
                data_piv.fillna("nnn", inplace=True)

                if data_piv.shape[0] < min_points:
                    print(
                        f"Threshold of points per transect {tr_id} not reached. It has {data_piv.shape[0]} points."
                    )
                    unreliable_trs.append(tr_id)

                else:
                    pass

            elif mode == "drop":
                data_piv.dropna(axis="index", how="any", inplace=True)
            else:
                raise NameError(" Specify the mode ('drop' or 'nnn')")

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
                steady_state["thresh"] = thresh
                steady_state["min_pts"] = min_points
                steady_state["valid_pts"] = n

                steady_state_tr = pd.concat(
                    [steady_state, steady_state_tr], ignore_index=True
                )

            except BaseException:
                print(f"tr_id {tr_id} has not enough points. Go ahead...")
                pass

        if unreal == "drop":

            print("eliminating unreliable transects . . . ")
            for i in unreliable_trs:

                indexa = steady_state_tr.query(f"tr_id =={i}").index
                steady_state_tr.drop(indexa, inplace=True)
        elif unreal == "keep":
            pass
        else:
            raise NameError(
                " Specify what to do with unreliable transects ('drop' or 'keep' ?)"
            )

    return steady_state_tr


def compute_rBCD_transects(
    dirNameTrans,
    steady_state_tr,
    loc,
    crs_dict_string,
    weights_dict,
    label_order=["ed", "hd", "md", "sd", "ud", "nnn", "ue", "se", "me", "he", "ee"],
    geo=True,
):
    """It computes transect-level r-BCDs, based on the steady-state dataframe returned by steady_state_transect function.

    Warning: changing label order is not supported as submatrix partitioning is hard-coded (TO UPDATe)

    Args:
        dirNameTrans (str): Full path to the folder where the transects are stored.
        steady_state_tr (df): dataframe containing the steady state distributions (returned by steady_state_transect function) of each transect.
        loc (str): Location code of the location being analysed. Used to match transect file.
        crs_dict_string (dict): Dictionary storing location codes as key and crs information as values, in dictionary form.
        Example: crs_dict_string = {'wbl': {'init': 'epsg:32754'},
                   'apo': {'init': 'epsg:32754'},
                   'prd': {'init': 'epsg:32755'},
                   'dem': {'init': 'epsg:32755'} }
        weigth_dict (dict): dictionary containing each magnitude class as keys and value to be used to weigth each class as values.
        This can be manually set or objectively returned by the infer_weights function (reccommended).
        label_order (list): order to arrange the states in the first-order and steady-state matrices.
        geo (bool): wether or not to return a Geopandas geodataframe (Default) instead of a Pandas dataframe.

    Returns:
       Two dataframes. The first is the transect-level r-BCDs (geo)dataframe and the second is a dataframe, useful for plotting purposes.
    """
    weigths_series = pd.Series(weights_dict, weights_dict.keys(), name="weight")

    ss_transects_idx = pd.DataFrame()
    loc_codes = [loc]

    for loc in loc_codes:

        sub = steady_state_tr.query(f"location=='{loc}'")
        sub = sub.pivot(index="markov_tag", columns="tr_id", values=("p"))
        try:
            sub.drop("nnn", inplace=True)
        except BaseException:
            pass

        # Apply the weights
        sub_w = sub.join(weigths_series, how="inner")
        weighted_matrix = sub_w.iloc[:, :-1].multiply(sub_w.weight, axis="index")
        weighted_matrix = weighted_matrix.loc[label_order, :]

        try:
            weighted_matrix.drop("nnn", inplace=True)
        except BaseException:
            pass

        # Create erosion and deposition sub-matrix
        erosion = weighted_matrix.iloc[5:, :].transpose()
        erosion["erosion"] = erosion.sum(axis=1)

        deposition = weighted_matrix.iloc[:-5, :].transpose()
        deposition["deposition"] = deposition.sum(axis=1)

        # Compute erosion residuals
        indexes_vic = abs(deposition.deposition) - abs(erosion.erosion)
        indexes_vic = pd.DataFrame(pd.Series(indexes_vic, name="residual"))
        indexes_vic.reset_index(inplace=True)
        indexes_vic.rename({"index": "tr_id"}, axis=1, inplace=True)

        # Put all into one table
        deposition["erosion"] = erosion.erosion
        to_plot = deposition.reset_index()[["index", "deposition", "erosion"]].rename(
            {"index": "tr_id"}, axis=1
        )

        to_print_table = to_plot.merge(indexes_vic, on="tr_id", how="left")
        to_plot = to_print_table.melt(
            id_vars=["tr_id"], var_name="process", value_name="coastal_index"
        )

        to_plot["location"] = loc

        coastal_markov_trs_steady = attach_trs_geometry(
            to_plot, dirNameTrans, list_loc_codes=loc_codes
        )

        piv_markov_trs = pd.pivot_table(
            coastal_markov_trs_steady,
            index=["location", "tr_id"],
            columns="process",
            values=["coastal_index"],
        ).reset_index(col_level=1)

        piv_markov_trs.columns = piv_markov_trs.columns.droplevel(0)
        trs_markov_idx = attach_trs_geometry(
            piv_markov_trs, dirNameTrans, list_loc_codes=loc_codes
        )

        ss_transects_idx = pd.concat(
            [trs_markov_idx, ss_transects_idx], ignore_index=True
        )

        if bool(geo):
            ss_transects_idx = gpd.GeoDataFrame(
                ss_transects_idx, geometry="geometry", crs=crs_dict_string[loc]
            )
        else:
            pass

    return ss_transects_idx, to_plot

def compute_multitemporal (df,
                            geometry_column="coordinates",
                           filter_sand=True,
                           date_field='survey_date',
                          sand_label_field='label_sand',
                           filter_classes=[0]):
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

                if filter_sand:
                    df_pre=loc_data.query(f"{date_field} =='{date_pre}' & {sand_label_field} in {filter_classes}").dropna(subset=['z'])
                    df_post=loc_data.query(f"{date_field} =='{date_post}' & {sand_label_field} in {filter_classes}").dropna(subset=['z'])
                else:
                    df_pre=loc_data.query(f"{date_field} =='{date_pre}'").dropna(subset=['z'])
                    df_post=loc_data.query(f"{date_field} =='{date_post}'").dropna(subset=['z'])

                merged=pd.merge(df_pre,df_post, how='inner', on='spatial_id', validate="one_to_one",suffixes=('_pre','_post'))
                merged["dh"]=merged.z_post.astype(float) - merged.z_pre.astype(float)

                dict_short={"geometry":merged.filter(like=geometry_column).iloc[:,0],
                            "location":location,
                            "tr_id":merged.tr_id_pre,
                            "distance":merged.distance_pre,
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
