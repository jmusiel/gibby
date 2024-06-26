import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.ticker import FuncFormatter


def plot_corrections_histogram(
        pred_dataframe,
        true_dataframe,
        name=None, 
        size=16
    ):
    """
    Plot a histogram of the corrections from a dataframe containing corrections.
    Args:
        pred_dataframe: ML predictions from hessian_to_corrections
        true_dataframe: VASP results from hessian_to_corrections
        name: for title of figure (optional)
        size: default 16
    Returns:
        None
    """
    pass

    fig, axs = plt.subplots(1, 4, figsize=(28, 6))
    fig.suptitle(f"{name} Gibbs Energy Correction Term Errors", fontsize=size)
    for i, (correction_key, correction_name) in enumerate([
        ("ts", "Entropy correction"),
        ("zpe", "Zero point energy correction"),
        ("deltah", "Heat capacity correction"),
        ("total", "Total correction"),
    ]):
        ml_values = pred_dataframe[correction_key].values
        vasp_values = true_dataframe[correction_key].values
        errors = np.abs(ml_values - vasp_values)
        errors_greater_than_threshold = len(errors[errors > 0.1])/len(errors)
        axs[i].hist(errors, bins=100, color='tab:blue', alpha=1, density=True, label=f"MAE: {np.mean(np.abs(errors)):.3f} [eV] -- percent >0.1: {100*errors_greater_than_threshold:.3f}%")
        axs[i].set_xlabel(f"{correction_name} abs err [eV]", fontsize=size)
        axs[i].legend()

        print(f"{correction_name.ljust(30)} -- MAE: {np.mean(np.abs(errors)):.3f} -- percent >0.1: {100*errors_greater_than_threshold:.3f}")

    fig.patch.set_facecolor('white')
    return fig


def plot_hexbin_corrections(
        pred_dataframes_list,
        true_dataframe,
        value_name: str = "eigenvalues",
        title_name=None, 
        size=16,
        pred_dataframes_names: list=None,
    ):
    """
    Plot a hexbin plot of the corrections from a dataframe containing corrections.
    Args:
        pred_dataframe: ML predictions from hessian_to_corrections
        true_dataframe: VASP results from hessian_to_corrections
        name: for title of figure (optional)
        size: default 16
    Returns:
        None
    """
    if value_name == "eigenvalues":
        get_values = get_eigenvalues
        xlabel = "eigenvalues ($eV/\AA^2$)"
        ylabel = "DFT eigenvalues ($eV/\AA^2$)"
    elif value_name == "total":
        get_values = get_total_corrections
        xlabel = "total correction [eV]"
        ylabel = "DFT total correction [eV]"
    elif value_name == "freq":
        get_values = get_frequencies
        import matplotlib.ticker as ticker
        @ticker.FuncFormatter
        def major_formatter(x, pos):
            label = f"{-x:.0f}i" if x < 0 else f"{x:.0f}"
            return label
        xlabel = "largest imaginary frequency ($cm^{-1}$)"
        ylabel = "DFT corresponding frequency ($cm^{-1}$)"

    fig_cols = 4
    if len(pred_dataframes_list) < 4:
        fig_cols = len(pred_dataframes_list)
    fig_rows = int(math.ceil(len(pred_dataframes_list)/fig_cols))

    fig, axs = plt.subplots(fig_rows, fig_cols, figsize=(8*fig_cols+4, 6))
    if title_name is not None:
        fig.suptitle(title_name, fontsize=size)

    axs = axs.flatten()
    for i, df in enumerate(pred_dataframes_list):
        i_name = "ML"
        if pred_dataframes_list is not None:
            i_name = pred_dataframes_names[i]

        ml_values = get_values(df)
        vasp_values = get_values(true_dataframe)

        min_max = (min(min(ml_values), min_max[0]), max(max(ml_values), min_max[1]))
        min_max = (min(min(vasp_values), min_max[0]), max(max(vasp_values), min_max[1]))

        hexbin0 = axs[i].hexbin(ml_values, vasp_values, gridsize=100, cmap='viridis', vmin=1, vmax=5, mincnt=1)
        axs[i].set_xlabel(f"{i_name} {xlabel}", fontsize=size)
        axs[i].set_ylabel(ylabel, fontsize=size)
        axs[i].set_aspect('equal', 'box')
        axs[i].set_xlim(min_max)
        axs[i].set_ylim(min_max)

        if value_name == "freq":
            axs[i].xaxis.set_major_formatter(major_formatter)
            axs[i].yaxis.set_major_formatter(major_formatter)

    # Create a formatter function that formats numbers as integers
    formatter = FuncFormatter(lambda x, pos: f"{x:.0f}")
    # add shared colorbar
    cbar = fig.colorbar(hexbin0, ax=axs.ravel().tolist(), format=formatter)
    # set colorbar ticks fontsize
    cbar.ax.tick_params(labelsize=size)
    # cbar.set_ticks([1, 2, 3, 4, 5])

    return fig

def get_eigenvalues(given_df):
    eigen_values_list = [sorted(eig) for eig in given_df["eigenvalues"].values]
    result = np.concatenate(eigen_values_list)
    result = np.real(result)
    return result

def get_total_corrections(given_df):
    result = given_df["total"].values
    return result

def get_frequencies(given_df):
    unsorted_freq_list = [freq for freq in given_df["freq"].values]
    sorted_freq_list = []
    for freq_sublist in unsorted_freq_list:
        sorted_args = np.argsort(np.array(freq_sublist)**2)
        sorted_freq_sublist = freq_sublist[sorted_args]
        sorted_freq_list.append(sorted_freq_sublist)
    values_list = []
    for freq_sublist in sorted_freq_list:
        leftmost = freq_sublist[0]
        real = np.real(leftmost)
        imag = np.imag(leftmost)
        value = real - imag
        values_list.append(value)
    return values_list