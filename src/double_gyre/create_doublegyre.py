from datetime import timedelta
from opendrift.readers import reader_double_gyre
from opendrift.models.oceandrift import OceanDrift
from src.double_gyre.dispersion_utils import exportParticlesToDataset

# PARAMETERS
A = 0.1
epsilon = 0.25
NB_PARTS = 25000
altSeed = 3

o = OceanDrift(loglevel=20)  # Set loglevel to 0 for debug information
o.fallback_values["land_binary_mask"] = 0
o.set_config("drift:scheme", "runge-kutta4")

double_gyre = reader_double_gyre.Reader(epsilon=epsilon, omega=6.28 / 10, A=A)
print(double_gyre)

o.add_reader(double_gyre)

x = [0.95]
y = [0.5]
lon, lat = double_gyre.xy2lonlat(x, y)

o.seed_elements(
    lon,
    lat,
    radius=0.1,
    number=NB_PARTS + altSeed,
    time=[double_gyre.initial_time, double_gyre.initial_time + timedelta(seconds=50)],
)

o.run(duration=timedelta(seconds=250), time_step=0.1)
# o.animation(buffer=0)
print(o.get_lonlats()[0].shape)

lons = o.get_lonlats()[0] * 30 / o.get_lonlats()[0].max() + 195
lats = o.get_lonlats()[1] * 20 / o.get_lonlats()[1].max() + 20

exportParticlesToDataset(
    "double_gyre_as_" + str(altSeed),
    lons[:NB_PARTS, 500:],
    lats[:NB_PARTS, 500:],
    epsilon,
    A,
)
