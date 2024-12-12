import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray


class XRDPlotter:
    X_LABEL = "2θ [°]"
    Y_LABEL = "Intensity [a.u.]"

    def __init__(
        self,
        two_thetas: NDArray,
        intensities: NDArray,
        line_color="k",
        use_seaborn: bool = False,
    ):
        self.two_thetas = two_thetas
        self.intensities = intensities
        self.line_color = line_color
        if use_seaborn:
            sns.set_style("darkgrid")

    def plot(self) -> tuple[Figure, Axes]:
        fig, ax = plt.subplots()
        ax.plot(self.two_thetas, self.intensities, color=self.line_color)
        ax.set_xlabel(self.X_LABEL)
        ax.set_ylabel(self.Y_LABEL)
        return fig, ax


if __name__ == "__main__":
    import numpy as np

    two_thetas = np.linspace(0, 90, 9000)
    intensities = np.random.rand(9000)
    plotter = XRDPlotter(two_thetas, intensities, use_seaborn=False)
    plotter.plot()
    plt.show()
