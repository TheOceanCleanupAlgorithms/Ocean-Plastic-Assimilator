# Data requirements

The assimilator can assimilate observations taken either:

- Sampled from another simulation, see section 1.a
- Recorded in a csv file, see section 1.b

In both cases, the simulation in which the observations will be assimilated has to be stored as a netCDF file containing particles data from a dispersion model.

#### Dimensions

| Name | Data Type | Description |
| --- | --- | --- |
| p_id | integer | particles indexes |
| time | integer | time indexes |

#### Variables

| Name | Dimensions | Data Type | Description |
| --- | --- | --- | --- |
| p_id | (p_id) | integer | numeric id of particle (coordinate variable) |
| lon | (p_id, time) | numeric | longitudes of particle through time |
| lat | (p_id, time) | numeric | latitudes of particle through time |
| time | (time) | integer | list of indexes of successive timesteps |

The longitudes and latitudes can be whatever you want, as long as they fit in the domain you give as input to the `run_assimilator` function.

## 1.a Sample from another simulation

In that case, the other simulation has to be stored with the very same specifications as the first one.

## 1.b Recorded in a csv file

In the case, the csv data must contain the following columns:

| Name | Data Type | Description |
| --- | --- | --- |
| lon_id | integer | observation longitude in the grid sent to the assimilator |
| lat_id | integer | observation latitude in the grid sent to the assimilator |
| value | float | value measured at the coordinates |
| variance | float | variance of the value measured at the coordinates |
| time | integer | time of the observation |

