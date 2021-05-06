from src.run_assimilator import run_assimilator

if __name__ == "__main__":
    run_assimilator(
        particles_dataset_path="data/advector_output_rivers_2012.nc",
        observations_type="from_csv",
        observations_source_path="data/observations.csv",
        assimilation_domain_coords=(195 - 360, 23, 235 - 360, 45),
        assimilation_grid_size=(int(40 / 0.5), int(22 / 0.5)),
        size_ensemble=10,
        initial_ensemble_spread=0.05,
        t_start=0,
        t_end=300,
        radius_observation=50,
    )
