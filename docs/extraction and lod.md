# Methods for elevation analysis

## Virtual network of elevation profiles

A virtual network of digital transects is used and kept fixed for every site during the analysis. Transects can be any line, digitised in a GIS, of irregular shape, spacing and lengths. Alternatively, the function `create_transects()` allows an user to create alongshore uniformingly distributed transects, normal to an user-provided shoreline.
This function can be useful in case a very long shoreline must be monitored in multiple sites.


## Data extraction

Elevation (from DSMs) and color (from orthophotos) data is extracted automatically using `ProfileSet.extract_profiles()` method. The parameter `sampling_step` indicates the distance (in meters) to use to create a sampling point along each transect. Those points will then extract the pixel values of each DSM and orthophoto projected below them, forming the initial, uncleaned dataframe of data.

## Limits of detection (LoD) thresholds

The morphological method ([Lane et al., 2003](https://onlinelibrary.wiley.com/doi/10.1002/esp.483)) is being increasingly used to monitor geomorphic changes and estimating sediment budgets in a variety of environments, including sandy beaches ([Gonçalves and Henriques, 2015](https://www.sciencedirect.com/science/article/abs/pii/S0924271615000532?via%3Dihub)). It normally involves the subtraction of two Digital Surface Model (DSM) representing the same location at different times to extract the elevation difference over a shared elevation datum. In this paper, cross-shore elevation profiles are subject to the same error estimation techniques used for raster-based (DSM) analysis. Thus, in order to avoid including regions of apparent changes due to DSM noise into volumetric calculations, a Limit of Detection (LoD) threshold per DoD must be quantified. Any elevation change (Δh) below the LoD is considered uncertain due to the magnitude of Δh inherent to DSMs noise. LoDs are typically derived solely from the elevation components of the DSM, as the horizontal positional errors have negligible influence when volumes in low slope areas, such as beaches, have to be calculated ([Wheaton, 2009](https://onlinelibrary.wiley.com/doi/10.1002/esp.1886)). When in the study areas pseudo-invariant (within the temporal scale of the DoD) features are present, such as roads, boulders, roofs, beach accesses and other anthropogenic constructions, these are usable as calibration zones and a simple LoD approach can be used. By calculating LoDs from the modelled DSMs we incorporate intrinsic (SfM-MVS pipeline, camera radial distortion, GCP accuracy, interpolation) and extrinsic (surface texture, illumination, topographic effects) errors that propagated throughout the data acquisition/processing chain.
The method used in Sandpyper derives LoDs that are relevant for each time period in every surveyed location.

LoD statistics are computed when the `ProfileDynamics.compute_multitemporal()` method is called, if, when extracting profiles with the `ProfileSet.extract_profiles()` method a directory storing the transects to use to extract LoD was provided with the `lod_mode` parameter. In that case, the LoD statistics are stored in the `ProfileDynamics.lod_df` attribute, which include for each survey:

1. mean (column: 'mean')
2. median (column: 'med')
3. standard deviation (column: 'std')
4. normalised medain absolute deviation (column: 'nmad')
5. 68.3th quantile of absolute error (column: 'a_q683')
6. 95th quantile of absolute error (column: 'a_q95')
7. robust root mean squared error (column: 'rrmse')
8. number of total observations (column: 'n')
9. n_outliers using the 3-sigma method (column: 'n_outliers')
10. Shapiro–Wilk statistics, p-values and normality evaluation (columns: 'saphiro_stat', 'saphiro_p', 'saphiro_normality')
11. D’Agostino–Pearson statistics, p-values and normality evaluation (columns: 'ago_stat', 'ago_p', 'agoo_normality')
12. the chosen Limit of Detection based on normality check (column: 'lod')

To choose which metric to use as LoD, given that LoD can be interpreted as the expected error in the DoD (DSM of Difference, basically the subtraction of two DSMs representing the landscape in two consecutive times), it is necessary to evaluate the normality of the statistical distribution of errors (values of changing elevations in calibration areas which are supposed not to be changing). Despite the Root Mean
Squared Error (rmse) being the most used error metric in the literature ([Carrivick, Smith and Quincey, 2016](https://play.google.com/store/books/details?id=tZCvDAAAQBAJ))
, its validity is robust only when a normal distribution of absolute errors with no outliers is assumed, which is seldom occurring due to filtering and interpolation errors introduced by the digital photogrammetric procedure ([Höhle and Höhle, 2009](https://www.sciencedirect.com/science/article/abs/pii/S0924271609000276?via%3Dihub)). The normalised median absolute deviation (nmad) is reported to be a more robust estimator for elevation precision of photogrammetric products, in case the above mentioned assumptions are not met ([[Höhle and Höhle, 2009](https://www.sciencedirect.com/science/article/abs/pii/S0924271609000276?via%3Dihub)], [Wang, Shi and Liu, 2015](https://www.sciencedirect.com/science/article/abs/pii/S0303243414001767?via%3Dihub)).


## LoD visual normality evaluation

In addition to performing statistical tests (Shapiro-Whilk and D’Agostino-Pearson tests),
Sandpyper allows to evaluated the normality of the absolute error distribution by visually assessing their Q-Q plots, as recommended in [D’Agostino et al. (1990)](tandfonline.com/doi/abs/10.1080/00031305.1990.10475751) and [Höhle and Höhle (2009)](https://www.sciencedirect.com/science/article/abs/pii/S0924271609000276?via%3Dihub), by calling the `ProfileDynamics.plot_lod_normality_check()`.
