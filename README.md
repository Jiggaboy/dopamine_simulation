# dopamine_simulation

## How to use it

Clone a branch.
An initial configurations is (usually) set up and the first run can be started without further fine tuning.

First of all, a connectivity matrix has to be created. Run 'python -m connectivity_matrix' in order to do so.

Next, you can simulate the network according to the configuration using the command 'python -m model_looper'. Ensure that the same configuration is defined in both files since these steps could be independent from each other.


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

## Programming advice
### Logging
'import cflogging' in the main file. It is a custom/configured logger setup. Run 'cflogger.set_up()'. Then you will have the same logger configuration in each module. 
Within a module 'import logging' and get a logger with 'logging.get_logger()'

