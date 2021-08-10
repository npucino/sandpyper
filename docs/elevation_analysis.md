# Methods for elevation analysis

## Virtual network of elevation profiles

A virtual network of digital transects is used and kept fixed for every site during the analysis. Transects can be any line, digitised in a GIS, of irregular shape, spacing and lengths. Alternatively, the function `create_transects()` allows an user to create alongshore uniformingly distributed transects, normal to an user-provided shoreline.
This function can be useful in case a very long shoreline must be monitored in multiple sites.


## Data extraction

Elevation (from DSMs) and color (from orthophotos) data is extracted automatically using `ProfileSet.extract_profiles()` method. The parameter `sampling_step` indicates the distance (in meters) to use to create a sampling point along each transect. Those points will then extract the pixel values of each DSM and orthophoto projected below them, forming the initial, uncleaned dataframe of data.

## Limits of detection (LoD) thresholds

The morphological method ([Lane et al., 2003](https://onlinelibrary.wiley.com/doi/10.1002/esp.483)) is being increasingly used to monitor geomorphic changes and estimating sediment budgets in a variety of environments, including sandy beaches ([Gonçalves and Henriques, 2015](https://www.sciencedirect.com/science/article/abs/pii/S0924271615000532?via%3Dihub)). It normally involves the subtraction of two Digital Surface Model (DSM) representing the same location at different times to extract the elevation difference over a shared elevation datum. In this paper, cross-shore elevation profiles are subject to the same error estimation techniques used for raster-based (DSM) analysis. Thus, in order to avoid including regions of apparent changes due to DSM noise into volumetric calculations, a Limit of Detection (LoD) threshold per DoD must be quantified. Any elevation change (Δh) below the LoD is considered uncertain due to the magnitude of Δh inherent to DSMs noise. LoDs are typically derived solely from the elevation components of the DSM, as the horizontal positional errors have negligible influence when volumes in low slope areas, such as beaches, have to be calculated ([Wheaton, 2009](https://onlinelibrary.wiley.com/doi/10.1002/esp.1886)). When in the study areas pseudo-invariant (within the temporal scale of the DoD) features are present, such as roads, boulders, roofs, beach accesses and other anthropogenic constructions, these are usable as calibration zones and a simple LoD approach can be used. By calculating LoDs from the modelled DSMs we incorporate intrinsic (SfM-MVS pipeline, camera radial distortion, GCP accuracy, interpolation) and extrinsic (surface texture, illumination, topographic effects) errors that propagated throughout the data acquisition/processing chain.
The method used in Sandpyper derives LoDs that are relevant for each time period in every surveyed location.


















- abs_in = Total rising
- abs_out = Total lowering
- abs_net_change= Net elevation change
- mec_m: Net elevation change/along. Beach length
- norm_in: abs_in/total number of points used in the calculation of this time period (n_obs_valid)
- norm_out: abs_out/total number of points used in the calculation of this time period (n_obs_valid)
- norm_net_change=abs_net_change/total number of points used in the calculation of this time period (n_obs_valid)
- tot_vol_depo: Total volume deposited
- tot_vol_ero: Total volume eroded
- net_vol_change: Net volume change
- location_m3_m: Net volume change / beach length
- n_obs_valid: Total number of valid observations, within beachface and beyond limits of detection
- cum: Cumulative net volume change since the beginning of the monitoring
- cum_mec: Cumulative norm_net_change since the beginning of the monitoring
