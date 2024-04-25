# dopamine_simulation

## How to use it

Clone a branch.
An initial configurations is (usually) set up and the first run can be started without further fine tuning.

First of all, a connectivity matrix has to be created. Run 'python -m connectivity_matrix' in order to do so.

Next, you can simulate the network according to the configuration using the command 'python -m model'. Ensure that the same configuration is defined in both files since these steps are independent from each other.


## Useful Links
### brian2
See: https://brian2.readthedocs.io/


## Terminology

Transfer function (tf): The accumulated input is transferred to a output rate according to a sigmoidal function.

## Landscape
Possible modes are:
 - Random: random preferred direction, but still with the specified shift.
 - Homogeneous: All neurons share the same preferred direction.
 - symmetric: No shift (thus no preferred direction) is considered.
 - Perlin: Preferred direction is assigned to (normal) Perlin noise.
 - Perlin_uniform: the Perlin noise is binned into 8 directions of equal size.
 - Independent: The synaptic connections are drawn across the population independent of preferred direcion and shift.


## Reproducibility of simulations
With the brian2 framework, the reference used is:
https://brian2.readthedocs.io/en/stable/advanced/random.html?highlight=random

In order to set the same seed for baseline vs patch simulations, one must create the network first and then set the brian2 internal seed.


## Programming advice
### Logging
'from cflogging import logger' initializes a (singleton) logger.
Configuration of the logger is changed in _cflogging.py_.

## History
### Sequence correlation using Gaussian kernel
Idea: Replace all spikes at two location with some Gaussian kernel. Then calculate the maximum (normalized) correlation, and find the corresponding time lag.

_Removed in v0.4_

### PCA
Idea: Check the correlation structure in a broader range than just the NM patch.

_Removed in v0.1 as a PCA does not add information, the activity is at a 1-D intrinsic manifold (discussion with Mark Humphries)_
