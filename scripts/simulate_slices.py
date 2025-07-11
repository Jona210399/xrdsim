from pathlib import Path

from pymatgen.core.structure import Structure
import pandas as pd
from tqdm import tqdm
from xrdsim.calculator import get_default_numpy_xrd_calculator, XRDCalculator
import numpy as np


def simulate_10k():
    tqdm.pandas()

    path = Path("/p/project1/hai_solaihack/datasets/slices/mattext_as_matbind.json")

    materials_df = pd.read_json(path)

    num_structures_to_simulate = 1000

    materials_df = materials_df.sample(n=num_structures_to_simulate, random_state=42)

    xrd_calculator = get_default_numpy_xrd_calculator()

    def structure_dict_to_xrd(structure_dict: dict, xrd_calculator: XRDCalculator):
        structure = Structure.from_dict(structure_dict)
        xrd_intensities = xrd_calculator.calculate(structure).tolist()
        return xrd_intensities

    materials_df["pxrd"] = materials_df["crystal_structure"].progress_apply(
        lambda x: structure_dict_to_xrd(x, xrd_calculator)
    )

    materials_df.to_parquet("mattext_as_matbind_with_xrds_10k.parquet")


def simulate_all():
    tqdm.pandas()
    path = Path("/p/project1/hai_solaihack/datasets/slices/mattext_as_matbind.json")
    num_chunks = 20
    start_chunk = 14

    print("Loading Data")

    materials_df = pd.read_json(path)

    xrd_calculator = get_default_numpy_xrd_calculator()

    def structure_dict_to_xrd(structure_dict: dict, xrd_calculator: XRDCalculator):
        structure = Structure.from_dict(structure_dict)
        xrd_intensities = xrd_calculator.calculate(structure).tolist()
        return xrd_intensities

    chunks = np.array_split(materials_df, num_chunks)
    chunks = chunks[start_chunk:]

    for i, chunk in enumerate(chunks, start=start_chunk):
        print(f"Processing Chunk {i}/{num_chunks}")
        chunk["pxrd"] = chunk["crystal_structure"].progress_apply(
            lambda x: structure_dict_to_xrd(x, xrd_calculator)
        )

        chunk.to_json(f"mattext/mattext_as_matbind_with_xrds_chunk_{i}.json")


def json_to_parquet_polars():
    tqdm.pandas()

    path = Path.cwd() / "mattext"
    start_at = 0
    stop_at = 20

    indices = list(range(stop_at))
    indices = indices[start_at:]
    print("Chunks", indices)

    import polars as pl
    import psutil

    dfs = []

    print(f"Available MEM: {psutil.virtual_memory()}")

    for i in indices:
        print(f"Processing {i}")
        print("Memory", psutil.virtual_memory().percent)
        name = f"mattext_as_matbind_with_xrds_chunk_{i}"
        file = path / f"{name}.json"
        df = pd.read_json(file)
        df = df.drop(columns=["crystal_structure", "slices", "cif_p1"])
        df = pl.from_pandas(
            df, schema_overrides={"material_id": pl.String, "pxrd": pl.List(pl.Float32)}
        )
        print("Appending")
        print(df)
        dfs.append(df)

    print("Concatenating")
    df: pl.DataFrame = pl.concat(dfs, how="vertical")
    print(df)
    print("Saving")
    save_path = path / "mattext_pxrds.parquet"

    df.write_parquet(save_path)
    print("DF Saved successfully")
    print("Reading DF")
    df = pl.read_parquet(save_path)
    print(df)


if __name__ == "__main__":
    # simulate_10k()
    # simulate_all()
    json_to_parquet_polars()
