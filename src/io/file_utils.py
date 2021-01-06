from pathlib import Path


def create_folder(dir_path: str):
    dir = Path(dir_path)
    if not dir.is_dir():
        dir.mkdir()
    else:
        print(
            f"Directory{dir_path}could not be created. It may already exist, and in that case Assimilator will not override the folder",
        )
        exit()
    print("Successfully created", dir_path)