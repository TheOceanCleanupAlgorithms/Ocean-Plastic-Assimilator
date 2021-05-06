from src.run_assimilator import run_assimilator
import numpy as np

cell_area = lambda dlon,dlat,lat: dlat * 111 * dlon * 111 * np.cos(lat * np.pi / 180)

GRID_COORDS = (-165, 23, -125, 45)
GRID_RESOLUTION = 0.5

cells_area = np.array([[cell_area(0.5, 0.5, lat_id * GRID_RESOLUTION + GRID_COORDS[1]) for lat_id in range(44)] for lon_id in range(80)])

if __name__ == "__main__":
    run_assimilator(
        particles_dataset_path="data/advector_output_rivers_2012.nc",
        observations_type="from_simulation",
        observations_source_path="data/advector_output_coastal_2012.nc",
        assimilation_domain_coords=GRID_COORDS,
        assimilation_grid_size=(int(40 / GRID_RESOLUTION), int(22 / GRID_RESOLUTION)),
        size_ensemble=10,
        initial_ensemble_spread=50,
        observations_error_percent=0.01,
        observation_locations=[(12, 4), (55, 27)],
        t_start=0,
        t_end=365,
        radius_observation=np.inf,
        initial_mass_multiplicator=1000, # Set to your initial particles mass
        cells_area=cells_area
    )
