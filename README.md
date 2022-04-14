# dopamine_simulation

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

