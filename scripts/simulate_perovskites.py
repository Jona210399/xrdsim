import json
from pathlib import Path

from pymatgen.core.structure import Structure
from tqdm import tqdm

from xrdsim.calculator import get_default_numpy_xrd_calculator


def simulate_sunmat_perovskites():
    path = Path.cwd().parent.parent
    with open(path.joinpath("inorganic_SUNMAT_10k.json"), "r") as file:
        data = json.load(file)

    xrd_calculator = get_default_numpy_xrd_calculator()
    for i in tqdm(range(len(data)), desc="Calculating XRD"):
        item = data[i]
        structure = Structure.from_dict(item["crystal_structure"])
        _, xrd_intensities = xrd_calculator.calculate(structure)
        item["xrd"] = xrd_intensities.tolist()

    with open("results/inorganic_SUNMAT_10k_with_xrds.json", "w") as file:
        json.dump(data, file)


def simulate_materials_project_perovskites():
    path = Path.cwd().parent.parent
    with open(path.joinpath("perovskite_data.json"), "r") as file:
        data = json.load(file)

    xrd_calculator = get_default_numpy_xrd_calculator()
    for i in tqdm(range(len(data)), desc="Calculating XRD"):
        item = data[i]
        structure = Structure.from_dict(item["crystal_structure"])
        _, xrd_intensities = xrd_calculator.calculate(structure)
        item["xrd"] = xrd_intensities.tolist()

    with open("results/perovskite_data_with_xrds.json", "w") as file:
        json.dump(data, file)


def inspect_xrds():
    import matplotlib.pyplot as plt
    import numpy as np

    with open("results/perovskite_data_with_xrds.json", "r") as file:
        data = json.load(file)

    x = np.linspace(5, 90, 8501)

    for i in range(10):
        plt.plot(x, data[i]["xrd"])
        plt.savefig(f"results/xrds/perovskite_{i}.png")
        plt.clf()


def extract_single_structure():
    path = Path.cwd().parent.parent
    with open(path.joinpath("perovskite_data.json"), "r") as file:
        data = json.load(file)

    structure_to_extract: int = 9

    with open("results/single_perovskite.json", "w") as file:
        json.dump(data[structure_to_extract], file)


def plot_dos():
    import matplotlib.pyplot as plt

    with open("results/single_perovskite.json", "r") as file:
        data = json.load(file)

    dos = data["density_of_states"]
    energies = dos["energies"]
    densities = dos["densities"]["1"]

    plt.plot(energies, densities)
    plt.savefig("results/dos.png")


if __name__ == "__main__":

    # simulate_materials_project_perovskites()
    # inspect_xrds()
    # extract_single_structure()
    # plot_dos()
    # simulate_sunmat_perovskites()

    pass
