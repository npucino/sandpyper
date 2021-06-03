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
affiliations:
    - name: School of Life and Environmental Sciences, Deakin University, Warrnambool, 3280, Australia
      index: 1
date: 6 June 2021
bibliography: paper.bib
---

# Summary

**sandpyper** is a Python package for 

# Intro

Coastal zones host 40% of the world population (Martínez et al., 2007)  and it is increasing, especially in least developed countries (Neumann et al., 2015). Sandy beaches, amongst other ecoservices (Barbier et al., 2011), protect inland assets from coastal erosion, dissipating stormy waves energy on their shores. Mitigating beach erosion typically involves the establishment of topographic monitoring programs in key locations (erosional hotspots) to quantify beach dynamics, erosion/deposition volumes, recovery times and model coastal resilience or risk to erosion. High temporal and spatial resolution topographic data is ideal, but expensive with most of the ordinary beach surveying methods.
Unmanned Aerial Vehicles (UAVs) and Structure from Motion algorithms (UAV-SfM) are emerging as the best platform and methodology to obtain cost-effective high-quality beach topographic data (as Digital Surface Models, DSMs) (Gonçalves and Henriques, 2015) at the mesoscale, a spatiotemporal resolution appropriate for coastal management (Thom et al., 2018). Consequently, researchers already use UAV-SfM to monitor beach dynamics around the world, but it has been limited so far to a few sites and a few multitemporal dates. However, UAV-SfM technology is mature and reliable enough to allow wider-scale and longer-term monitoring projects.
For instance in Victoria (Australia), a citizen-science UAV-SfM monitoring program mobilises more than 150 volunteers to fly UAVs on 15 sites every six weeks for three years. To date, volunteers flew 350 times, enabling the creation of a DSM and an orthophoto per survey (uncompressed file sizes from 5-10 Gb each), which generates an unprecedented archive of imagery which can be reliably used to monitor high-frequency sandy beach volumetric dynamics and behaviors (Pucino et al., 2021).

# Statement of Need

A drawback of using UAV-SfM for beach monitoring is that due to UAV regulations, flight altitude is often limited to around 80-120 m above ground, which means that the ground sampling distance of consumer-grade UAVs is sub-decimeter, resulting in very high resolution and large imagery files, especially for beach surveys exceeding the 20 ha coverage. Although managing tens of large rasters with geographic information systems such as Qgis or ESRI ArcGIS is technically feasible, handling tens to hundreds of such files within large monitoring projects quickly becomes impractical.
Moreover, in coastal management, erosion assessments from multitemporal DSMs is usually approached by raster subtraction (also known as dem of difference method, see Lane, James and Crowell, 2000), which is a process to compute elevation difference from time to time by subtracting the elevation value of each cell in the two pre and post rasters. Raster-based operations with full-resolution UAV-SfM imagery becomes very time consuming with important computing power and memory needs that can.
Therefore, tradeoffs for working within a GIS could include raster spatial downsampling, which might cause losing important information about equally important smaller scale geomorphological landforms (Walker et al., 2017), or, tiling the rasters into smaller and more manageable units, which ultimately further increases total pre-processing time.
Furthermore, beach-specific challenges are (1) the water motion as waves wash in and out of the swash zone, which prevents SfM algorithm from modelling elevation accurately, (2) dune vegetation and (3) stranded beach wracks (macroalgae, woody debris), which should be removed or filtered as these can bais sediment volumetric computation.

Sandpyper is an open-source Python package that provides a user with tools specifically designed to overcome the aforementioned limitations, from the generation of cross-shore transects and extraction of colour and elevation information from a collection of rasters, to the analysis and plotting of beachface cluster dynamics indices. It offers users the possibility to perform volumetric and behavioural monitoring of beaches in a programmatic way, enabling them to chain functions one after the other or use them separately as standalone processing steps. Moreover, by using a naming convention, it allows to manage multiple locations with different coordinate reference systems.


Currently v0.1.1 allows to:

* automatically create user-defined georeferenced cross-shore transects along a line and extract elevation (from DSMs) and colour (from orthophotos) profiles.
* use unsupervised machine learning and user-provided polygon masks to clean the profiles from unwanted non-sand points and swash zones.
* compute altimetric and volumetric timeseries analysis and plotting the results (Fig 1, 2).
* use spatial autocorrelation measures to discard spatial outliers and obtain statistically significant Hotspots/Coldspots areas of beach change at the site and transect scales.
* compute Beachface Cluster Dynamics indices (Pucino et al., 2021) at the site and transect scales.

Moreover, a module called “space” is under development, which adds some functionalities to the study of beach dynamics combining UAV-SfM surveys with spaceborne-derived shorelines. Some features already in Sandpyper are:
* spatial grid generation along a line (waterline, shoreline).
* custom tiling of georeferenced imagery, including multispectral satellite imagery, UAV orthomosaics or single band images (DSMs, label masks).
* waterline to shoreline simple tidal correction.
* waterline /shoreline error assessments.
* shoreline shift assessment.

Sandpyper is aimed at being further developed to be a wider-scope package as its functions can be applied to any scope involving the extraction of information from a large amount of rasters.

# Usage

Various tutorials and documentation are available for using **sandpyper**, including:

-   [Jupyter notebook examples for using sandpyper](https://github.com/npucino/sandpyper/tree/master/examples)
-   [Complete documentation on sandpyper modules and functions](https://npucino.github.io/sandpyper/)

# Acknowledgements

Funding provided by Deakin University and the Victorian Department of Environment, Land, Water and Planning.

# References
