
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


<!-- PROJECT logo -->
<br />
<p align="center">
  <a href="https://github.com/npucino/sandpyper/">
    <img src="images/sandpiper1-01.png" alt="Logo" width="50%" height="50%"> 
  </a>


  <h3 align="center">- Tools for automatic UAV-SfM beach volumetric and behavioural analysis -</h3>

  <p align="justify">
    Sandpyper performs an organised and automatic extraction of elevation profiles from as many DSM and orthophotos as you like. It is thought to be used when a considerable number of DSMs and orthohpotos from many different locations and coordinate reference systems need to be processed.
    Then, computes volumetric and behavioural analysis of beachfaces, speeding up an otherwise long and difficult to handle (big rasters) job.
    It has some specialised functions to deal with the common limitations found in beach environments:
  <ol>
    <li>Swash zone: the water motion as waves wash in and out of the swash zone prevent Structure from Motion algorithm to find matches, thus, model elevation.</li>
    <li>Vegetation: both dune vegetation and beach wracks (macroalgae, woody debris) should be removed or filetered as can compromise sediment volumetric computation.</li>
    <li>File size: a few km long beach surveyed with a Phantom 4-Advanced at 100m altitude create roughly 10 Gb (uncompressed) of data, which can be cumbersome for some GIS to handle.</li>
  </ol>

  From user-defined cross-shore transects, you can clean profiles from unwanted non-sand points, detect significant hotspots/coldspots (cluster) of beach change, compute       volumetric dynamics at the site and transect levels, plot alongshore change and model beachface behaviour using the Beachface Cluster Dynamics indices.

  Plus, some outils functions that can come at hand, like automatic transect creation from a vector line, grid creation along a line and subsequent tiles extraction and others.

  Follow the Jupyter Notebook tutorials (IN PREPARATION) to understand how it works!

  >**This code has supported the analysis and publication of the article ["Citizen science for monitoring seasonal-scale beach erosion and behaviour with aerial drones"](https://rdcu.be/cfgvu  "link to paper"), in the open access Nature Scientific Report journal.**


  <br />
  <a href="https://github.com/npucino/sandpyper"><strong>Explore the docs »</strong></a>
  <br />
  <a href="https://github.com/npucino/sandpyper">View Demo</a>
  ·
  <a href="https://github.com/npucino/sandpyper/issues">Report Bug</a>
  ·
  <a href="https://github.com/npucino/sandpyper/issues">Request Feature</a>
  </p>
</p>



<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#Background">Background</a></li>
        <li><a href="#Modules">Modules</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#Installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#publications">Publications</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>

  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

<img src="images/banner3.png" alt="banner" width="100%" height="50%">

<!-- Background -->
### Background
This code has been developed to ease the analysis of big data coming from more than 300 Unnmanned Aerial Vehicles (UAV) surveys, performed by Citizen Scientist in Victoria, Australia. This is the [Eureka Award Winning](https://www.deakin.edu.au/about-deakin/media-releases/articles/eureka-prize-win-for-world-first-citizen-science-program) World-first [beach monitoring program powered by volunteers](https://www.marinemapping.org/vcmp-citizen-science "info on the Citizen Science project"), who autonomously fly UAVs on 15 sensitive sites (erosional hotspots) across the Victorian coast, every 6 weeks for 3 years, since 2018. This project is part of a broader marine mapping program called The [Victorian Coastal Monitoring Program (VCMP)](https://www.marineandcoasts.vic.gov.au/coastal-programs/victorian-coastal-monitoring-program "VCMP website"), funded by the [Victorian Department of Environment, Land, Water and Planning](https://www.delwp.vic.gov.au/ "DELWP website"), co-funded by [Deakin University](https://www.deakin.edu.au/ "Deakin Uni website") and [The University of Melbourne](https://www.unimelb.edu.au/ "UniMelb website").

Each survey creates Digital Surface Models and orthophotos of considerable size (5-10 Gb uncompressed), which can be troublesome for some GIS to render, let alone perform rster-based computations.

<!-- modules -->
### Modules

* **.outils**: main storage of useful functions.
* **.profile**: extract elevation and colour information across profiles from a folder of rasters.
* **.labels**: use KMean and Silhouette analysis to help clean the extracted elevation data from unwanted non-sand points.
* **.hotspot**: use Local Moran's I to disscard spatial outliers and obtain significant hot/cold spots of beach change at site and transect levels.
* **.dynamics**: compute Beachface CLuster Dynamics indices at the site and transect levels.
* **.volumetrics**: compute volumetric timeseries of sand-only points and create plots.
* **.space** (under development): simple satellite and UAV instantaneous waterlines tidal-correction and shoreline shifts/error metrics.



<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites
* [Install Conda] in your local machine. We need it to create the **"sandpyper_env"** virtual environment.
* then, if you do not have it already, install **Visual Studio C++ build tools** in your local machine, which is needed to install the [richdem](https://pypi.org/project/richdem/ "richdem pypi page") package. You can download it [here](https://visualstudio.microsoft.com/visual-cpp-build-tools/ "standalone VS c++ build tools").
* If you don't have it already, add __conda-forge__ channel to your anaconda config file, by typing this in your Anaconda Prompt terminal (base environment):
  ```sh
  conda config --add channels conda-forge
  ``` 
* Now, always in the (base) environment, create a new environment called **sandpyper_env** and install the required packages by typing:
  ```sh
  conda create --name sandpyper_env geopandas=0.8.2 matplotlib=3.3.4 numpy=1.20.1 pandas=1.2.2 tqdm pysal=2.1 rasterio=1.2.0 richdem=0.3.4 scikit-image=0.18.1 scikit-learn=0.24.1 scipy=1.6 seaborn=0.11.1 tqdm=4.56.2
  ```
* If __rasterio__ package cannot be installed due to __GDAL binding issues__, follow the instructions in [rasterio installation webpage](https://rasterio.readthedocs.io/en/latest/installation.html).

### Installation

1. ```sh
   conda activate sandpyper_env
   ```
2. ```sh
   pip install sandpyper
   ```
2. Install Jupyter Notebooks:
   ```sh
   conda install jupyter notebook
   ```
3. Once you open a Jupyter Notebook with the sandpyper_env, import it to test it works.
   ```sh
   import sandpyper
   ```

<!-- USAGE EXAMPLES -->
## Usage

TO DO.

_For more examples, please refer to the [Documentation](https://npucino.github.io/sandpyper/)_



<!-- ROADMAP -->
## Roadmap

See the [open issues](https://github.com/npucino/sandpyper/issues) for a list of proposed features (and known issues).



<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Nicolas Pucino - [@NicolasPucino](https://twitter.com/@NicolasPucino) - npucino@deakin.edu.au

Project Link: [https://github.com/npucino/sandpyper](https://github.com/npucino/sandpyper)


<!-- Publications -->
## Publications

* [Pucino, N., Kennedy, D. M., Carvalho, R. C., Allan, B. & Ierodiaconou, D. Citizen science for monitoring seasonal-scale beach erosion and behaviour with aerial drones. Sci. Rep. 11, 3935 (2021)](https://rdcu.be/cfgvu)

<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

* [Funding provided by Deakin University and the Victorian Department of Environment, Land, Water and Planning](https://www.marineandcoasts.vic.gov.au/coastal-programs/victorian-coastal-monitoring-program)
* [Deakin University Citizen Science program webpage](https://www.marinemapping.org/vcmp-citizen-science)
* [A/Prof Daniel Ierodiaconou](https://www.deakin.edu.au/about-deakin/people/daniel-ierodiaconou), my PhD supervisor and the [Deakin Marine Mapping team](https://www.marinemapping.org/)






<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
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

[Install Conda]:https://www.anaconda.com/products/individual "Anaconda download website"
