## Volumetrics

The exclusion of points within the swash extent and variations in UAV survey coverage often results in irregularities in the number of transects and valid points per survey. Therefore, to compare subaerial changes across sites and time, Sandpyper computes the mean elevation change (MEC), as follows:

$$
\operatorname{Mean\,Elevation\,Change\,(MEC)} = \frac{1}{n}\mathop \sum \limits_{z = 0}^{n} (z_{post} - z_{pre} )
$$

where $n$ is the total number of valid elevation points, $z_{pre}$ and $z_{post}$ are the elevations above the height datum used occurring at the same location in both pre and post surveys.

Additionally, when no inter-site comparisons is involved, Sandpyper allows to approximate the alongshore volumetric change (in m3/m) as:

$$
\operatorname{Along.\,beachface\,change} = \mathop \smallint \limits_{{x_{swash} }}^{{x_{limit} }} \left( {z_{post} - z_{pre} } \right)dx
$$

where $x_{swash}$ and $x_{limit}$ are the upper swash and landward limit respectively. Plus or minus ($\pm$) error intervals for both MEC and volumetric change estimates represent the uncertainty related to changes within the limit of detection thresholds.



## Volumetrics tables details

When the `ProfileDynamics.compute_volumetrics()` is called, two new attributes storing two volumetric dataframes are added:
1. `ProfileDynamics.location_volumetrics`: volumetrics at the location level.
2. `ProfileDynamics.transects_volumetrics`: volumetrics at the transect level.

Both dataframe contains the same columns, which are here explained:

- abs_in = Total rising of points (meters)
- abs_out = Total lowering of points (meters)
- abs_net_change= Net elevation change, abs_in - abs_out (meters)
- mec_m= Net elevation change/alongshore beach length (computed as number of transects * transect spacing) (m/m)
- norm_in= abs_in/total number of points used in the calculation of this time period (n_obs_valid)
- norm_out= abs_out/total number of points used in the calculation of this time period (n_obs_valid)
- norm_net_change=abs_net_change/total number of points used in the calculation of this time period (n_obs_valid)
- tot_vol_depo = Total volume deposited (m3)
- tot_vol_ero = Total volume eroded (m3)
- net_vol_change= Net volume change (m3)
- location_m3_m = Net volume change / beach length (m3)
- n_obs_valid = Total number of valid observations, within beachface and beyond limits of detection
