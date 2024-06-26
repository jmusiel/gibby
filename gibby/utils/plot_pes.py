import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import numpy as np
import matplotlib
import scipy

def get_pes(
    ax: matplotlib.axes.Axes,
    E: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    grid_length:int = 100,
    gaussian_filter_stdev:float = 1,
    gaussian_filter_order:int = 0
    ):
    """
    Plot a potential energy surface (PES) with a contour plot.
    Args:
        ax: The axes object to plot the PES on.
        E: The potential energy values.
        x: The x values.
        y: The y values.
        grid_length: The number of points to interpolate the PES onto. Default is 100.
        gaussian_filter_stdev: The standard deviation of the Gaussian filter to apply to the
            PES. Default is 1.
        gaussian_filter_order: The order of the Gaussian filter to apply to the PES. Default is 0.

    Returns:
        None
    """
    E_rev = [Ei for idx, Ei in enumerate(E) if not np.isnan(x[idx])]
    x_rev = [xi for xi in x if not np.isnan(xi)]
    y_rev = [yi for yi in y if not np.isnan(yi)]

    total_length = 100
    x1 = np.linspace(min(x_rev), max(x_rev), total_length)
    y1 = np.linspace(min(y_rev), max(y_rev), total_length)
    grid_x, grid_y = np.meshgrid(x1, y1)
    z1 = griddata(np.transpose(np.array([x_rev,y_rev])), E_rev, (grid_x, grid_y), method='linear')
    z1  = scipy.ndimage.gaussian_filter(z1,1)
    img = plt.imshow(z1, cmap='RdBu_r')
    ax.set_xlabel('x [arbitrary units]')
    ax.set_ylabel('y [arbitrary units]')
    
    cbar = plt.colorbar(img)
    cbar.set_label('Energy [eV]')
