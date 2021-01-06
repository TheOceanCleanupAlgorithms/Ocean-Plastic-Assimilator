import netCDF4 as nc
from glob import glob

varmap = {"id": "p_id", "x": "p_id"}
files_to_change = glob("data/parts_double_gyre_*.nc")

if __name__ == "__main__":
    print(f"Running update script with varmap {varmap}")
    for f in files_to_change:
        print(f"Updating file {f}")
        ds = nc.Dataset(f, "a")
        for old_key in varmap:
            try:
                ds.renameVariable(old_key, varmap[old_key])
                print(f"Renamed var {old_key} to {varmap[old_key]}")
            except KeyError:
                pass

            try:
                ds.renameDimension(old_key, varmap[old_key])
                print(f"Renamed dim {old_key} to {varmap[old_key]}")
            except KeyError:
                pass
        ds.close()
