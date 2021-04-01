# Welcome to Sandpyper !


[![image](https://img.shields.io/pypi/v/sandpyper.svg)](https://pypi.python.org/pypi/sandpyper)
[![Contributors][contributors-shield]][contributors-url]
[![image](https://github.com/npucino/sandpyper/workflows/build/badge.svg)](https://github.com/npucino/sandpyper/actions/workflows/build.yml/badge.svg)
[![image](https://github.com/npucino/sandpyper/workflows/docs/badge.svg)](https://github.com/npucino/sandpyper/actions/workflows/docs.yml/badge.svg)
[![image](https://github.com/npucino/sandpyper/workflows/pypi/badge.svg)](https://github.com/npucino/sandpyper/actions/workflows/pypi.yml/badge.svg)
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]


**- Tools for automatic UAV-SfM beach volumetric and behavioural analysis -**


-   GitHub repo: <https://github.com/npucino/sandpyper/tree/master/sandpyper>
-   Documentation: <https://npucino.github.io/sandpyper/>
-   PyPi: <https://pypi.org/project/sandpyper/>
-   Notebook examples: <https://github.com/npucino/sandpyper/tree/master/examples>
-   Free software: MIT license

## Introduction

**Sandpyper** is an open-source Python package that allows a user to perform UAV-SfM volumetric and behavioural monitoring of sandy beaches in an automated and efficient way.


## Key Features

-  create user-defined georeferenced cross-shore transects along a line
-  extract elevation (from DSMs) and color (from orthophotos) information from folders containing hundreds of rasters
-  use unsupervised machine learning and user-provided polygon masks to clean the profiles from unwanted non-sand points and swash zone
-  compute altimetric and volumetric timeseries analysis
-  use spatial autocorrelation measures to discard spatial outliers and obtain statistically significant Hotspots/Coldspots areas of beach change at the site scale
- compute Beachface Cluster Dynamics indices (Pucino et al., 2021) at the site and transect scales
- compute limits of detections
- plot and visualise beach change and dynamics

Additionally, a new module called **space** is being developed to groudtruth satellite-derived shorelines with UAV-derived shorelines.
Currently, it allows:

- spatial grid generation along a line (waterline, shoreline)
- extraction and export of tiles (from the grid) of multispectral satellite imagery
- waterline to shoreline simple tidal correction
- waterline /shoreline error assessments


## Credits

This package was created with [Cookiecutter](https://github.com/cookiecutter/cookiecutter) and the [giswqs/pypackage](https://github.com/giswqs/pypackage) project template.








[contributors-shield]: https://img.shields.io/github/contributors/npucino/sandpyper.svg?style=image
[contributors-url]: https://github.com/npucino/sandpyper/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/npucino/sandpyper.svg?style=image
[forks-url]: https://github.com/npucino/sandpyper/network/members
[stars-shield]: https://img.shields.io/github/stars/npucino/sandpyper.svg?style=image
[stars-url]: https://github.com/npucino/sandpyper/stargazers
[issues-shield]: https://img.shields.io/github/issues/npucino/sandpyper.svg?style=image
[issues-url]: https://github.com/npucino/sandpyper/issues
[license-shield]: https://img.shields.io/github/license/npucino/sandpyper.svg?style=image
[license-url]: https://github.com/npucino/sandpyper/blob/master/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=image&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/nicolaspucino/
