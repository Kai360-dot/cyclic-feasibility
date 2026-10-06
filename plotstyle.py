"""Shared matplotlib settings for the thesis notebooks.

    from plotstyle import setup, FIGURES_DIR
    setup()

"""

from pathlib import Path
import matplotlib.pyplot as plt

# Figures are saved into a local clone of the GitHub repository of the thesis:
FIGURES_DIR = Path(r"C:\Users\kr326\Documents\MScThesis\figures")


def setup(font_size: float = 14, markerscale: float = 1.5, dpi: int = 300, retina: bool = True) -> None:
    """Apply the notebook plot settings for the rest of the session.

    font_size    rcParams font.size (ticks, labels and legends follow it)
    markerscale  legend.markerscale: legend markers are drawn this many times larger than in the
                 plot, so that s=2..8 scatter points stay visible in the legend;
    dpi          savefig.dpi;
    retina       high-resolution inline figures
    """
    plt.rcParams.update({
        "font.size": font_size,
        "legend.markerscale": markerscale,
        "savefig.dpi": dpi,
        "savefig.bbox": "tight",
    })
    if retina:
        try:
            from matplotlib_inline.backend_inline import set_matplotlib_formats
        except ImportError:  # plain Python, nothing to configure
            return
        set_matplotlib_formats("retina")
