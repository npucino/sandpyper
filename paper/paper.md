---
title: "sandpyper: A Python package for UAV-SfM beach volumetric and behavioural analysis"
tags:
    - Python
    - beach
    - uav-imagery
    - mapping
    - shoreline
authors:
    - name: Nicolas Pucino
      orcid: 0000-0001-6260-6647
      affiliation: "1"
    - name: David M. Kennedy
      orcid: 0000-0002-4878-7717
      affiliation: "2"
    - name: Daniel Ierodiaconou
      orcid: 0000-0002-7832-4801
      affiliation: "1"
affiliations:
    - name: School of Life and Environmental Sciences, Deakin University, Warrnambool, 3280, Australia
      index: 1
    - name: School of Geography, The University of Melbourne, Melbourne, 3010, Australia
      index: 2
date: 8 June 2021
bibliography: paper.bib
---

# Summary

**Sandpyper** is a Python package that automates profile-based volumetric and altimetric sandy beaches analysis from a large amount of digital surface models and orthophotos.  It includes functions to facilitate the cleaning of the elevation data from unwanted non-sand points or swash areas (where waves run up on the beach slope and 3D reconstruction is inaccurate) and to model beachface behavioural regimes using the Beachface Cluster Dynamics indices.

# Intro

Coastal zones host 40% of the world population [@Martinez2007-ux]  and it is increasing, especially in least developed countries [@Neumann2015-ph]. Sandy beaches, amongst other ecoservices [@Barbier2011-qp], protect inland assets from coastal erosion, dissipating stormy waves energy on their shores. Mitigating beach erosion typically involves the establishment of topographic monitoring programs in key locations (erosional hotspots) to quantify beach dynamics, erosion/deposition volumes, recovery times and model coastal resilience or risk to erosion. High temporal and spatial resolution topographic data is ideal, but expensive with most of the ordinary beach surveying methods.
Unmanned Aerial Vehicles (UAVs) and Structure from Motion algorithms (UAV-SfM) are emerging as the best platform and methodology to obtain cost-effective high-quality beach topographic data (as Digital Surface Models, DSMs) [@Goncalves2015-qr] at the mesoscale, a spatiotemporal resolution appropriate for coastal management [@Thom2018-kf]. Consequently, researchers already use UAV-SfM to monitor beach dynamics around the world, but it has been limited so far to a few sites and a few multitemporal dates. However, UAV-SfM technology is mature and reliable enough to allow wider-scale and longer-term monitoring projects.
For instance in Victoria (Australia), a citizen-science UAV-SfM monitoring program mobilises more than 150 volunteers to fly UAVs on 15 sites every six weeks for three years. To date, volunteers flew 350 times, enabling the creation of a DSM and an orthophoto per survey (uncompressed file sizes from 5-100 Gb each), which generates an unprecedented archive of imagery which can be reliably used to monitor high-frequency sandy beach volumetric dynamics and behaviors [@Pucino2021-ox].

# Statement of Need

A drawback of using UAV-SfM for beach monitoring is that due to UAV regulations, flight altitude is often limited to around 80-120 m above ground, which means that the ground sampling distance of consumer-grade UAVs is sub-decimeter, resulting in very high resolution and large imagery files, especially for beach surveys exceeding the 20 ha coverage. Although managing tens of large rasters with geographic information systems such as Qgis or ESRI ArcGIS is technically feasible, handling tens to hundreds of such files within large monitoring projects quickly becomes impractical.
Moreover, in coastal management, erosion assessments from multitemporal DSMs is usually approached by raster subtraction (also known as dem of difference method, see [@Lane2000-gs]), which is a process to compute elevation difference from time to time by subtracting the elevation value of each cell in the two pre and post rasters. Raster-based operations with full-resolution UAV-SfM imagery becomes very time consuming with important computing power and memory needs that can limit their feasibility.
Therefore, tradeoffs for working within a GIS could include raster spatial downsampling, which might cause losing important information about equally important smaller scale geomorphological landforms [@Walker2017-iw], or, tiling the rasters into smaller and more manageable units, which ultimately further increases total pre-processing time.
Furthermore, beach-specific challenges are (1) the water motion as waves wash in and out of the swash zone, which prevents SfM algorithm from modelling elevation accurately, (2) dune vegetation and (3) stranded beach wracks (macroalgae, woody debris), which should be removed or filtered as these can bias sediment volumetric computation.

**Sandpyper** is an open-source Python package that provides users with a processing pipeline specifically designed to overcome the aforementioned limitations, from the generation of cross-shore transects and extraction of colour and elevation information from a collection of rasters, to the analysis of period-specific limits of detection and plotting of beachface cluster dynamics indices. It offers users the possibility to perform volumetric and behavioural monitoring of beaches in a programmatic and organised way at the location and single transect scale. Moreover, by using a naming convention, it allows to manage multiple locations with different coordinate reference systems. Although originally developed for coastal areas, **Sandpyper** can be applied in many other environments where DSMs and orthophotos timeseries are used to monitor changes, such as river levee, glacier or gully monitoring.

Some previous works that are somehow related to **Sandpyper** include [Pybeach](https://github.com/TomasBeuzen/pybeach)[@Beuzen2019], a tool to automate beach dune toe identification and the [Digital Shoreline Analysis System (DSAS)](https://www.usgs.gov/centers/whcmsc/science/digital-shoreline-analysis-system-dsas?qt-science_center_objects=0#qt-science_center_objects), a tool to analyse shoreline shifts over time. While Pybeach is no longer maintained, the popularity of DSAS within the coastal erosion studies is fueled by its simple to use interface and the availability of a plug-in for [ESRI ArcMap](https://www.esri.com/en-us/arcgis/about-arcgis/overview) geographical information system. However, despite **Sandpyper**'s planned expansion to study satellite-derived-shorelines with a method inspired by DSAS, DSAS core objective is the study of horizontal shoreline migrations over time, with no functionalities in terms of three-dimensional profile extraction, volumetric and altimetric analysis or behavioural modeling.
To the best of the authors knowledge, this is the first Python package with the specific aim to integrate within an erosion monitoring project employing UAVs and SfM. Moreover, it is the only package which currently implements the BCDs.

Currently v0.1.1 allows to:

* automatically create user-defined georeferenced cross-shore transects (Figure 1a) along a line and extract elevation (from DSMs) and colour (from orthophotos) profiles.
* facilitate unsupervised machine learning sand classification (Figure 1b) and profile masking.
* compute altimetric and volumetric timeseries analysis and plotting the results, at the transect (Figure 1c) and site scales (Figure 1d).
* use spatial autocorrelation measures to discard spatial outliers and obtain statistically significant Hotspots/Coldspots areas of beach change at the site scale (Figure 2a).
* compute first-order transition probabilities of magnitude of change classes to derive Beachface Cluster Dynamics indices (Figure 2c) [@Pucino2021-ox].

Moreover, **Sandpyper** is being currently developed to include raster-based volumetric and behavioural analysis and satellite-derived-shorelines analysis. Some features already in **Sandpyper** are:

* custom spatial grid generation along a line (waterline, shoreline).
* custom tiling of georeferenced imagery, including multispectral satellite imagery, UAV orthomosaics or single band images (DSMs, label masks).
* shorelines tidal correction.
* shoreline error assessments in respect to groundtruth shorelines.
* shoreline shifts statistics.

**Sandpyper** is aimed at being further developed to be a wider-scope package as its functions can be applied to any scope involving the extraction of information from a large amount of rasters.

# Usage

Various tutorials and documentation are available for using **Sandpyper**, including:

-   [Jupyter notebook examples for using sandpyper](https://github.com/npucino/sandpyper/tree/master/examples)
-   [Complete documentation on sandpyper modules and functions](https://npucino.github.io/sandpyper/)

# Figures

## Figure 1
![**Example of the volumetric change computation pipeline.** (A) A sample virtual transects network. (B) Sand and no-sand classified points, facilitated with iterative Silhouette and KMeans analysis. (C) Alongshore transect-scale altimetric (top) and volumetric (bottom) change. (D) Site-level Mean Elevation Change (MEC) timeseries.\label{fig:ground}](joss_fig1.png)
**Example of the volumetric change computation pipeline.** (A) A sample virtual transects network. (B) Sand and no-sand classified points, facilitated with iterative Silhouette and KMeans analysis. (C) Alongshore transect-scale altimetric (top) and volumetric (bottom) change. (D) Site-level Mean Elevation Change (MEC) timeseries.

## Figure 2
![**Example of derivation of e-BCD indices.** (A) statistical significant clusters of elevation changes (hot-coldspots) timeseries. (B) First-order transition probabilities matrices. (C) e-BCDs plot derived.\label{fig:dists}](joss_fig2.png)
**Example of derivation of e-BCD indices.** (A) statistical significant clusters of elevation changes (hot-coldspots) timeseries. (B) First-order transition probabilities matrices. (C) e-BCDs plot derived.

# Acknowledgements

Funding provided by Deakin University and the Victorian Department of Environment, Land, Water and Planning.
Some functions in space module have been inspired and adapted from the [Digital Earth Australia github repository](https://github.com/GeoscienceAustralia/dea-notebooks) (credits in the docstrings).

# References
