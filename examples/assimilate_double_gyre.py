from src.run_assimilator import run_assimilator

mu = 2
A = 0.1175
epsilon = 0.25
sigma_rel = 0.01

if __name__ == "__main__":
    run_assimilator(
        particles_dataset_path="data/parts_double_gyre_standard_as_3.nc",
        observations_type="from_simulation",
        observations_source_path=f"data/parts_double_gyre_ref_eps_{epsilon}_A_{A}_as_2.nc",
        simulation_name=f"mu_{mu}_A_{A}_eps_{epsilon}_sigma_rel_{sigma_rel}",
        assimilation_domain_coords=(195, 20, 225, 40),
        assimilation_grid_size=(60, 40),
        size_ensemble=10,
        initial_ensemble_spread=0.05,
        t_start=0,
        t_end=2000,
        observations_error_percent=sigma_rel,
        observation_locations=[(12, 4), (55, 27)],
        initial_mass_multiplicator=mu,
    )
