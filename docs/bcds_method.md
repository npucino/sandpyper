# Beachface Cluster Dynamics indices

The Beachface Cluster Dynamics (BCD) indices are purposefully designed novel metrics to leverage the very high spatiotemporal resolutions and three-dimensionality of UAV SfM topographic data for quantifying subaerial beach morphodynamics. Transition matrices and steady-state probability vectors of Markov chain models representing Δh classes temporal dynamics are used to compute e-BCDs and r-BCD respectively.
The fundamental difference between the two types is that e-BCDs are computed based on dynamics observed over the monitoring period (represented in the transition matrices) while r-BCDs represent what one would expect the sediment budget to be when the same erosional or depositional dynamics are projected over an infinite time steps (steady-state).
Their fact that they contain the terms __beachface cluster__ in their names indicate that BCDs are originally conceived to characterize beachface sediment dynamics of hotspots (technically better named statistically significant clusters) of changes. However, in case of BCDs at transect-level, the scarcity of valid and usable points can lead to the decision to take all the points into account rather than only the hotspots.

Moreover, as the r-BCD to incorporates full 3D profile timeseries, this index can also be seen as a behavioural regime indicator, as it describes what was the underlying tendency of the system during the monitoring period.

## Discrete Markov chain

Following Lambin (1994), a discrete Markov process can be represented as:

$s_{t + 1} = Ms_{t}$

<br>

where <img src="https://render.githubusercontent.com/render/math?math=s_{t}"> is a column vector, <img src="https://render.githubusercontent.com/render/math?math=s=(s1,..., sm)"> having as elements the valid points (within the beachface and beyond LoD sand-only Δh points) in one of the <img src="https://render.githubusercontent.com/render/math?math=m"> states (i.e. Δh magnitude classes) at time <img src="https://render.githubusercontent.com/render/math?math=t">. <img src="https://render.githubusercontent.com/render/math?math=M"> is a <img src="https://render.githubusercontent.com/render/math?math=m X m"> matrix holding the first-order (from <img src="https://render.githubusercontent.com/render/math?math=t"> to <img src="https://bit.ly/3AtwyrO" align="center" border="0" alt="t+1 " width="36" height="15" /> transition probabilities <img src="https://render.githubusercontent.com/render/math?math=p_{ij}"> , derived as:

<img src="https://bit.ly/3CCjm60" align="center" border="0" alt="p_{ij} = \frac{{n_{ij} }}{{\mathop \sum \nolimits_{k = 1}^{m} n_{ik} }}," width="114" height="37" /><br>

where <img src="https://render.githubusercontent.com/render/math?math=n_{ij}"> is the number of transitions from an initial state <img src="https://render.githubusercontent.com/render/math?math=i"> to state <img src="https://render.githubusercontent.com/render/math?math=j"> and <img src="https://render.githubusercontent.com/render/math?math=m"> is the number of states (i.e. elevation change magnitude classes) in which each observation can be. The matrix <img src="https://render.githubusercontent.com/render/math?math=M"> is row-standardised, so that the sum of transition probabilities from a given state is always equal to one.

## Empirical Beachface Cluster Dynamics (e-BCDs)
The e-BCDs divide the transition matrix into four sub-matrices, each representing site-level erosional, depositional, recovery and vulnerability “behaviors” of the subaerial beach over the monitoring period (see image below).

<center> <img src="images/wbl_p_matrix.png" alt="Warrnambool submatrices" width="300"/></center>

The e-BCDs indices are computed for every sub-matrices (<img src="https://render.githubusercontent.com/render/math?math=m">) as follows:

<img src="https://bit.ly/3yGX7cK" align="center" border="0" alt="Empirical\, BCD_{sub} = \mathop \sum \limits_{i,j = 1}^{n} \left[ { ws^{\prime}_{ij} } \right]\times p_{ij} ," width="278" height="50" /><br>

where <img src="https://render.githubusercontent.com/render/math?math=p_{ij}"> is the probability of any significant cluster on the sandy beachface to transition from an initial state <img src="https://render.githubusercontent.com/render/math?math=i"> to the following state <img src="https://render.githubusercontent.com/render/math?math=j"> and <img src="https://render.githubusercontent.com/render/math?math=w_{ij}'"> is a weight that reflects the importance (severity) of the transitions in erosion and deposition contexts from a coastal planning perspective (higher weights to worst transitions, such as from Extreme Deposition to Extreme Erosion), which is defined as:

<img src="https://bit.ly/3yGXbJw" align="center" border="0" alt="ws^{\prime}_{ij} = \left\{ {\begin{array}{*{20}l} { ws_{i} \times\left( { - ws_{j} } \right) \leftrightarrow i > j } \\ {ws_{i} \times ws_{j} \leftrightarrow i \le j } \\ \end{array} } \right\}," width="258" height="42" /><br>

where <img src="https://render.githubusercontent.com/render/math?math=ws_{i}"> and <img src="https://render.githubusercontent.com/render/math?math=ws_{j}"> are the weights related to the initial <img src="https://render.githubusercontent.com/render/math?math=i"> and transitioning <img src="https://render.githubusercontent.com/render/math?math=j"> states respectively. The brackets “[ ]” indicate that the <img src="https://render.githubusercontent.com/render/math?math=ws'_{ij}"> transformation is implemented separately to determine the e-BCD sign only. The e-BCD absolute score computation does not implement this multiplication, capturing the process importance only. Any state to either no-hotspot or no-data transition probabilities are not included in the e-BCD interpretation.

## Residual Beachface Cluster Dynamics (r-BCDs)
The r-BCDs are computed from the steady-state probability vector.
The steady-state of a Markov chain returns a unique probability vector representing the states limiting probability distribution, which, once attained, one additional (or more) time steps will return the exact same initial states probabilities, signaling a situation of dynamic equilibrium has been achieved. This is represented as:

<img src="https://bit.ly/3lQXDRH" align="center" border="0" alt="\pi M = \pi ," width="67" height="17" /><br>

where <img src="https://render.githubusercontent.com/render/math?math=π_{j}">   is the vector containing the limiting probabilities <img src="https://render.githubusercontent.com/render/math?math=π"> for each <img src="https://render.githubusercontent.com/render/math?math=j">  state in <img src="https://render.githubusercontent.com/render/math?math=s"> . This vector <img src="https://render.githubusercontent.com/render/math?math=π">  is derived by solving a system of m equations with <img src="https://render.githubusercontent.com/render/math?math=m">  unknowns, each equation represented as:

<img src="https://bit.ly/3xH5Rhn" align="center" border="0" alt="\pi_{j} = \mathop \sum \limits_{k = 1}^{m} \pi_{k} p_{kj} ," width="108" height="47" /><br>

given that:

<img src="https://bit.ly/3iDPOgm" align="center" border="0" alt="\mathop \sum \limits_{j = 1}^{m} \pi_{j} = 1." width="79" height="50" />

The steady-state can be seen in a descriptive way as representing the states hierarchy, which is unique to the system being modelled ([Brown, 1970](http://dx.doi.org/10.2307/143152)), from which we derive the stochastic tendency the system had towards depositional or erosional states at the end of monitoring. We interpret this tendency as the most likely behavioural regime the system was subjected to, given the drivers of change and boundary conditions that influenced its evolution during the monitoring period.

The computation of the r-BCDs is as follows:
<img src="https://bit.ly/3lPQq4r" align="center" border="0" alt="Residual \,BCD_{SS} = 100 \times \mathop \sum \limits_{i = 1}^{m} (\pi_{ie} - \pi_{id} )," width="293" height="47" /><br>

where <img src="https://render.githubusercontent.com/render/math?math=ss">  is the steady-state probability distribution of one location, <img src="https://render.githubusercontent.com/render/math?math=π_{ie}">  are the limiting probabilities of the erosional classes (such as undefined erosion, small erosion, medium erosion, high erosion, extreme erosion) and <img src="https://render.githubusercontent.com/render/math?math=π_{id}">  the limiting probabilities of the depositional classes (undefined deposition, small deposition, medium deposition, high deposition and extreme deposition). The r-BCDs are not signed as no transitions are represented in the resultant vector. Any state to either no-hotspot or no-data transition probabilities are not included in the r-BCD interpretation. The multiplication by 100 is performed for index readability purposes.
