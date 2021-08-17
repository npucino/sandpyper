# Hotspot analysis

Spatial autocorrelation analysis uses defined statistics, such as Moran’s Index ([Moran, 1948](http://www.jstor.org/stable/2983777)), Getis and Ord’s G and G* ([Getis and Ord, 2010](https://onlinelibrary.wiley.com/doi/10.1111/j.1538-4632.1992.tb00261.x)) or Geary’s c ([Geary, 1954](https://www.jstor.org/stable/2986645?origin=crossref)) to summarise the observed spatial patterns of a variable of interests in the context of its spatial neighborhood. It tests and rejects the null-hypothesis of spatial randomness (<img src="https://render.githubusercontent.com/render/math?math=H_{0}">) by evaluating the probability that the occurrence of a spatial event of a certain intensity (i.e erosion or deposition) is independent to other spatial events that occur within the same neighbourhood. In this way, these spatial statistics can evaluate wether a statistically significant cluster is present in the data, which are commonly referred to as 'hotspots' and 'coldspots'.
In order to obtain spatially explicit and statistically significant clusters of erosion or deposition at the site level, the Local Moran-I (<img src="https://render.githubusercontent.com/render/math?math=I_{i}">) statistics with False Discovery Rate correction has been implemented in Sandpyper and is performed for every elevation difference (Δh) points (observations). The <img src="https://render.githubusercontent.com/render/math?math=I_{i}"> statistics is considered a Local Indicator of Spatial Association (LISA),  and is defined as:

![equation](https://latex.codecogs.com/gif.latex?Local%20Moran%5E%7B%5Cprime%7Ds%20Index%20%5Cleft%28%20%7B%20I_%7Bi%7D%20%7D%20%5Cright%29%20%3D%20%5Cfrac%7B%7Bz_%7Bi%7D%20-%20%5Cmu_%7Bz%7D%20%7D%7D%7B%7B%5Csigma%5E%7B2%7D%20%7D%7D%5Cmathop%20%5Csum%20%5Climits_%7Bj%20%3D%201%2C%20j%20%5Cne%20i%20%7D%5E%7Bn%7D%20%5Cleft%5B%20%7Bw_%7Bij%7D%20%5Cleft%28%20%7Bz_%7Bj%7D%20-%20%5Cmu_%7Bz%7D%20%7D%20%5Cright%29%7D%20%5Cright%5D%2C)



<img src="https://render.githubusercontent.com/render/math?math=e^{i \pi} = -1">

where <img src="https://render.githubusercontent.com/render/math?math=z_{i}"> is the value of the variable at location <img src="https://render.githubusercontent.com/render/math?math=i"> with <img src="https://bit.ly/2XiJy5l" align="center" border="0" alt=" \mu_{i} " width="19" height="12" /> and <img src="https://bit.ly/3yJtAza" align="center" border="0" alt=" \sigma^2  " width="19" height="17" /> the respective mean and variance, as calculated on the <img src="https://render.githubusercontent.com/render/math?math=n"> number of observations; <img src="https://render.githubusercontent.com/render/math?math=w_{ij}"> the spatial weight between the observation at location <img src="https://render.githubusercontent.com/render/math?math=i"> and <img src="https://render.githubusercontent.com/render/math?math=j"> and <img src="https://render.githubusercontent.com/render/math?math=z_{j}"> the value of the variable at all locations different than <img src="https://render.githubusercontent.com/render/math?math=i"> ([Anselin, 1995](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1538-4632.1995.tb00338.x)).

The conceptualization of space (the neighborhood) is done by computing a spatial weight matrix representing neighborhoods as spatial regions enclosing each observations. Sandpyper uses [Pysal](https://pysal.org/) in the backend and `ProfileDynamics.LISA_site_level()` method provides 3 different ways of creating the spatial weight matrix:

1. distance: uses a distance-band kernel and binary weights (1: neighbor, 0: no-neighbor)
2. k nearest neighbors: uses a specified number (k) of closest points to compute weights
3. Inverse distance weight (idw): weights decay as a function of distance to the focus point.

For instance, by using the distance weight matrix and specifying a 35 m distance value means that all the sand points within the 35 m band distance has been given a weight of 1, while the ones falling outside it received a weight of 0.


In order to evaluate pseudo-significance (pseudo p-values) of each Δh cluster a computational random permutation approach is used. An empirical statistical distribution of the <img src="https://render.githubusercontent.com/render/math?math=I_{i}"> statistic under <img src="https://render.githubusercontent.com/render/math?math=H_{0}"> (spatial randomness) assumption is created by randomly permuting (with no replacement) all the points within their neighborhood for 999 times. This distribution is then used to compare the observed <img src="https://render.githubusercontent.com/render/math?math=I"> statistic and decide if <img src="https://render.githubusercontent.com/render/math?math=H_{0}"> can be rejected in favor of <img src="https://render.githubusercontent.com/render/math?math=H_{1}"> (significant clustering, not due to spatial randomness) with a minimum pseudo p-value of 0.05 (95% level of confidence).

The significant <img src="https://render.githubusercontent.com/render/math?math=I_{i}"> can be placed in __4 different quadrants__ relative to the mean-centered Moran’s scatterplot, where each quadrant represent either an hotspot class or a spatial outlier, as:

* High-High (HH) clusters: areas where high values are surrounded by high values
* Low-High (LH) spatial outliers: low values among high values
* Low-Low (LL) clusters : areas where low values are surrounded by low values
* High-Low (HL) spatial outliers: high values among low values

For more information on LISAs, Ii and Moran’s scatterplot refer to [Anselin (1995)](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1538-4632.1995.tb00338.x) or watch Luc Anselin lecture below:

![im](images/luc_anselin.jpg)
[Watch here](https://www.youtube.com/watch?v=HF25odbiV3U&t=1109s)
<br>

In most use cases spatial outliers (LH and HL) are discarded, despite they represent interesting transition zones between the most and least affected areas. It is important to note that HH and LL clusters are relative to the mean of the whole dataset considered. Thus, counterintuitively, in an extreme case where erosion ( negative Δh ) has occurred on the totality of the dataset (either a whole survey or a specific transect), the HH clusters will represent higher negative values (closer to zero) indicating lower erosion magnitudes. On the other hand, LL clusters represent lower negative values (further from zero), indicating higher erosion magnitudes.
This is why Sandpyper discretizes Δh clusters and uses the created magnitude classes (extreme erosion, small deposition, etc.) as transition states for discrete Markov model creations and BCD computation.
